import torch
from core.utils.zoom_utils import deepim_boxes, batch_crop_resize
from core.utils.camera_geometry import bboxes_from_pose, centers_2d_from_pose, get_K_crop_resize
from .engine_utils import _empty, boxes_to_masks, get_input_dim, get_init_pose_train, _normalize_image
from .batch_test import batch_data_test, batch_updater_test


def batch_data(cfg, data, device="cuda", phase="train", dtype=torch.float32):
    if phase != "train":
        return batch_data_test(cfg, data, device=device, dtype=dtype)

    # batch training data and flatten
    tensor_kwargs = {"dtype": dtype, "device": device}
    to_float_args = {"dtype": dtype, "device": device, "non_blocking": True}
    to_long_args = {"dtype": torch.long, "device": device, "non_blocking": True}

    batch = {}
    num_imgs = len(data)
    batch["img"] = torch.stack([d["image"] for d in data]).to(**to_float_args)
    # construct flattened instance data =============================
    batch["obj_cls"] = torch.cat([d["instances"].obj_classes for d in data], dim=0).to(**to_long_args)
    batch["obj_bbox"] = torch.cat([d["instances"].obj_boxes.tensor for d in data], dim=0).to(**to_float_args)
    batch["obj_pose"] = torch.cat([d["instances"].obj_poses.tensor for d in data], dim=0).to(**to_float_args)
    batch["obj_visib_mask"] = torch.cat([d["instances"].obj_visib_masks.tensor for d in data], dim=0).to(
        **to_float_args
    )
    batch["obj_trunc_mask"] = torch.cat([d["instances"].obj_trunc_masks.tensor for d in data], dim=0).to(
        **to_float_args
    )
    # for PM loss (extent, points, sym_infos)
    batch["obj_extent"] = torch.cat([d["instances"].obj_extents for d in data], dim=0).to(**to_float_args)
    batch["obj_points"] = torch.cat([d["instances"].obj_points for d in data], dim=0).to(**to_float_args)

    num_insts_per_im = [len(d["instances"]) for d in data]
    num_insts_all = len(batch["obj_cls"])
    K_list = []
    sym_infos_list = []
    im_ids = []
    inst_ids = []  # the idx in the current img, should be used with im_ids
    for i_im in range(num_imgs):
        sym_infos_list.extend(data[i_im]["instances"].obj_sym_infos)
        for i_inst in range(num_insts_per_im[i_im]):
            im_ids.append(i_im)
            inst_ids.append(i_inst)
            K_list.append(data[i_im]["cam"].clone())

    batch["im_id"] = torch.tensor(im_ids, **tensor_kwargs)
    batch["inst_id"] = torch.tensor(inst_ids, **tensor_kwargs)
    batch["K"] = torch.stack(K_list, dim=0).to(**to_float_args)
    batch["sym_info"] = sym_infos_list

    # keep max_objs ----------------------------
    n_obj = min(cfg.DATALOADER.MAX_OBJS_TRAIN, num_insts_all)
    for _k in batch:
        if len(batch[_k]) == num_insts_all:
            batch[_k] = batch[_k][:n_obj]

    backbone_cfg = cfg.MODEL.DEEPIM.BACKBONE
    im_H, im_W = batch["img"].shape[-2:]
    h_in, w_in = backbone_cfg.INPUT_H, backbone_cfg.INPUT_W
    if backbone_cfg.INPUT_MASK:  # TODO: use box_ren
        batch["mask_obs"] = boxes_to_masks(batch["obj_bbox"], imH=im_H, imW=im_W, **tensor_kwargs)
        batch["zoom_mask_ren"] = _empty(n_obj, 1, h_in, w_in)

    batch["zoom_img_ren"] = _empty(n_obj, 3, h_in, w_in)
    input_dim = get_input_dim(cfg)
    if isinstance(input_dim, tuple):
        batch["zoom_x_obs"] = _empty(n_obj, input_dim[0], h_in, w_in)
        batch["zoom_x_ren"] = _empty(n_obj, input_dim[1], h_in, w_in)
    else:  # shared
        batch["zoom_x"] = _empty(n_obj, input_dim, h_in, w_in)

    if cfg.MODEL.DEEPIM.LOSS_CFG.FLOW_LW > 0 or backbone_cfg.INPUT_DEPTH:
        batch["zoom_flow"] = _empty(n_obj, 2, h_in, w_in)
        batch["zoom_depth_ren"] = _empty(n_obj, 1, h_in, w_in)  # gl ren depth
        if cfg.MODEL.DEEPIM.LOSS_CFG.FLOW_LW > 0:
            batch["zoom_depth_gl"] = _empty(n_obj, 1, h_in, w_in)  # gl obs depth  (not depth_obs !!!)

    return batch


def batch_updater(
    cfg,
    batch,
    renderer,
    poses_est=None,
    device="cuda",
    dtype=torch.float32,
    phase="train",
):
    if phase != "train":
        return batch_updater_test(
            cfg,
            batch,
            renderer,
            poses_est=poses_est,
            device=device,
            dtype=dtype,
        )
    # batch updater for train =========================================
    tensor_kwargs = {"dtype": dtype, "device": device}
    input_cfg = cfg.INPUT
    net_cfg = cfg.MODEL.DEEPIM
    backbone_cfg = cfg.MODEL.DEEPIM.BACKBONE
    loss_cfg = cfg.MODEL.DEEPIM.LOSS_CFG
    im_H, im_W = batch["img"].shape[-2:]
    h_in, w_in = backbone_cfg.INPUT_H, backbone_cfg.INPUT_W

    # for renderer
    image_tensor = torch.cuda.FloatTensor(h_in, w_in, 4, device=device).detach()  # image
    seg_tensor = torch.cuda.FloatTensor(h_in, w_in, 4, device=device).detach()  # mask
    pc_cam_tensor = torch.cuda.FloatTensor(h_in, w_in, 4, device=device).detach()  # depth
    pixel_mean = torch.tensor(cfg.MODEL.PIXEL_MEAN, **tensor_kwargs).view(3, 1, 1)
    pixel_std = torch.tensor(cfg.MODEL.PIXEL_STD, **tensor_kwargs).view(3, 1, 1)

    n_obj = batch["obj_pose"].shape[0]
    if poses_est is None:
        # init pose --------------------------------------------------------------------------
        get_init_pose_train(cfg, batch, **tensor_kwargs)  # obj_pose_est
    else:
        batch["obj_pose_est"] = poses_est
    # get crop bboxes and resize ratios based on obj_pose_est and obj_bbox
    ren_boxes = bboxes_from_pose(
        batch["obj_points"],
        K=batch["K"],
        pose=batch["obj_pose_est"],
        z_min=0.1,
    )
    ren_centers_2d = centers_2d_from_pose(K=batch["K"], pose=batch["obj_pose_est"], z_min=0.1)
    crop_bboxes, resize_ratios = deepim_boxes(
        ren_boxes,
        ren_centers_2d,
        obs_boxes=batch["obj_bbox"],
        lamb=input_cfg.ZOOM_ENLARGE_SCALE,
        imHW=(im_H, im_W),
        outHW=(h_in, w_in),
    )
    batch["zoom_K"] = get_K_crop_resize(batch["K"], crop_bboxes[:, :2], resize_ratios)  # for rendering zoomed data
    # get zoomed obs data ---------------------------------------------------
    im_rois = torch.cat([batch["im_id"].view(-1, 1), crop_bboxes], dim=1)  # crop and resize image_obs
    inst_rois = torch.cat([torch.arange(n_obj, **tensor_kwargs).view(-1, 1), crop_bboxes], dim=1)
    zoom_img_obs = batch_crop_resize(batch["img"], im_rois, out_H=h_in, out_W=w_in)
    if backbone_cfg.INPUT_DEPTH:
        # zoom_depth_obs = deepim_crop_resize(batch['depth_obs'], im_rois, out_H=h_in, out_W=w_in)
        raise NotImplementedError()
    if backbone_cfg.INPUT_MASK:  # crop and resize mask_obs
        zoom_mask_obs = batch_crop_resize(batch["mask_obs"], inst_rois, out_H=h_in, out_W=w_in)

    # get zoomed mask target
    if net_cfg.MASK_HEAD.ENABLED and loss_cfg.MASK_LW > 0:
        batch["zoom_visib_mask"] = batch_crop_resize(
            batch["obj_visib_mask"][:, None, :, :],
            inst_rois,
            out_H=h_in,
            out_W=w_in,
        )[:, 0, :, :]
        batch["zoom_trunc_mask"] = batch_crop_resize(
            batch["obj_trunc_mask"][:, None, :, :],
            inst_rois,
            out_H=h_in,
            out_W=w_in,
        )[:, 0, :, :]

    # yapf: disable
    # get zoomed ren data (directly render the zoomed data with zoomed K) ---------------
    for _i in range(n_obj):
        renderer.render(
            [int(batch["obj_cls"][_i])],
            [batch["obj_pose_est"][_i].detach().cpu().numpy()],  # src pose
            K=batch["zoom_K"][_i].detach().cpu().numpy(),
            image_tensor=image_tensor, seg_tensor=seg_tensor, pc_cam_tensor=pc_cam_tensor)
        batch["zoom_img_ren"][_i].copy_(image_tensor[:, :, :3].permute(2, 0, 1), non_blocking=True)  # CHW, 255
        if backbone_cfg.INPUT_MASK:
            batch["zoom_mask_ren"][_i, 0].copy_(
                (seg_tensor[:, :, 0] > 0).to(torch.float32), non_blocking=True)

        if cfg.MODEL.DEEPIM.LOSS_CFG.FLOW_LW > 0 or backbone_cfg.INPUT_DEPTH:
            batch["zoom_depth_ren"][_i, 0].copy_(pc_cam_tensor[:, :, 2], non_blocking=True)
            # render depth gl for obs
            if cfg.MODEL.DEEPIM.LOSS_CFG.FLOW_LW > 0:
                renderer.render(
                    [int(batch["obj_cls"][_i])],
                    [batch["obj_pose"][_i].detach().cpu().numpy()],  # tgt pose
                    K=batch["zoom_K"][_i].detach().cpu().numpy(),
                    # image_tensor=image_tensor, seg_tensor=seg_tensor,
                    pc_cam_tensor=pc_cam_tensor)
                batch['zoom_depth_gl'][_i, 0].copy_(pc_cam_tensor[:, :, 2], non_blocking=True)

    batch["zoom_img_ren"] = _normalize_image(batch["zoom_img_ren"], pixel_mean, pixel_std)  # normalize images

    if cfg.MODEL.DEEPIM.LOSS_CFG.FLOW_LW > 0:  # compute optical flow gt
        from core.csrc.flow import flow_torch
        zoom_flow, _ = flow_torch.flow(
                batch["zoom_depth_ren"], batch["zoom_depth_gl"],
                batch["obj_pose_est"], batch["obj_pose"], batch["zoom_K"])
        batch["zoom_flow"].copy_(zoom_flow, non_blocking=True)
    # yapf: enable

    # collect zoomed ren/obs data -----------------------------
    if backbone_cfg.SHARED:  # im_ren, im_obs
        batch["zoom_x"][:, 0:3] = batch["zoom_img_ren"]
        batch["zoom_x"][:, 3:6] = zoom_img_obs
        if backbone_cfg.INPUT_MASK:  # im_ren, im_obs, mask_ren, mask_obs
            batch["zoom_x"][:, 6:7] = batch["zoom_mask_ren"]
            batch["zoom_x"][:, 7:8] = zoom_mask_obs
            if backbone_cfg.INPUT_DEPTH:  # im_ren, im_obs, mask_ren, mask_obs, depth_ren, depth_obs
                # batch["zoom_x"][:, 8:9] = batch['zoom_depth_ren']
                # batch["zoom_x"][:, 9:10] = zoom_depth_obs
                raise NotImplementedError()
        else:  # not input mask
            if backbone_cfg.INPUT_DEPTH:  # im_ren, im_obs, depth_ren, depth_obs
                # batch['zoom_x'][:, 6:7] = batch['zoom_depth_ren']
                # batch['zoom_x'][:, 7:8] = zoom_depth_obs
                raise NotImplementedError()
    else:  # unshared inputs
        batch["zoom_x_ren"][:, :3] = batch["zoom_img_ren"]  # im_ren
        batch["zoom_x_obs"][:, :3] = zoom_img_obs  # im_obs
        if backbone_cfg.INPUT_MASK:
            batch["zoom_x_ren"][:, 3:4] = batch["zoom_mask_ren"]  # im_ren, mask_ren
            batch["zoom_x_obs"][:, 3:4] = zoom_mask_obs  # im_obs, mask_obs
            if backbone_cfg.INPUT_DEPTH:
                # batch['zoom_x_ren'][:, 4:5] = batch['zoom_depth_ren']  # im_ren, mask_ren, depth_ren
                # batch['zoom_x_obs'][:, 4:5] = zoom_depth_obs  # im_obs, mask_obs, depth_obs
                raise NotImplementedError()
        else:  # not input mask
            if backbone_cfg.INPUT_DEPTH:  #
                # batch['zoom_x_ren'][:, 3:4] = batch['zoom_depth_ren']  # im_ren, depth_ren
                # batch['zoom_x_obs'][:, 3:4] = zoom_depth_obs  # im_obs, depth_obs
                raise NotImplementedError()
    # done batch update train------------------------------------------
