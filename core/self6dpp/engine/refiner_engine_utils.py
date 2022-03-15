import os.path as osp
import random
import torch
import numpy as np
import itertools

from lib.dr_utils.DIBR.renderer_DIBR import Renderer_DIBR, load_ply_models
from core.utils.pose_aug import aug_poses_normal
from core.utils.pose_utils import rot_from_axangle_chain
from lib.pysixd.transform import random_rotation_matrix
from core.utils.zoom_utils import deepim_boxes, batch_crop_resize
from core.utils.camera_geometry import (
    bboxes_from_pose,
    centers_2d_from_pose,
    get_K_crop_resize,
)
from lib.vis_utils.image import grid_show, heatmap
from lib.vis_utils.optflow import flow2rgb


def get_init_pose_train(cfg, batch, device="cuda", dtype=torch.float32):
    tensor_kwargs = {"dtype": dtype, "device": device}
    input_cfg = cfg.INPUT
    n_obj = batch["obj_pose"].shape[0]
    init_pose_type = random.choice(input_cfg.INIT_POSE_TYPE_TRAIN)  # randomly choose one type
    if init_pose_type == "gt_noise":
        batch["obj_pose_est"] = aug_poses_normal(
            batch["obj_pose"],
            std_rot=input_cfg.NOISE_ROT_STD_TRAIN,  # randomly choose one
            std_trans=input_cfg.NOISE_TRANS_STD_TRAIN,  # [0.01, 0.01, 0.05]
            max_rot=input_cfg.NOISE_ROT_MAX_TRAIN,  # 45
            min_z=input_cfg.INIT_TRANS_MIN_Z,  # 0.1
        )
    elif init_pose_type == "canonical":
        r_canonical = rot_from_axangle_chain(input_cfg.CANONICAL_ROT)
        t_canonical = np.array(input_cfg.CANONICAL_TRANS)
        pose_canonical = torch.tensor(
            np.hstack([r_canonical, t_canonical.reshape(3, 1)]),
            **tensor_kwargs,
        )
        batch["obj_pose_est"] = pose_canonical.repeat(n_obj, 1, 1)  # [n,3,4]
    elif init_pose_type == "random":  # random
        poses_rand = np.zeros((n_obj, 3, 4), dtype="float32")
        for _i in range(n_obj):
            poses_rand[_i, :3, :3] = random_rotation_matrix()[:3, :3]
            t_min = input_cfg.RANDOM_TRANS_MIN
            t_max = input_cfg.RANDOM_TRANS_MAX
            poses_rand[_i, :3, 3] = np.array([random.uniform(_min, _max) for _min, _max in zip(t_min, t_max)])
        batch["obj_pose_est"] = torch.tensor(poses_rand, **tensor_kwargs)
    else:
        raise ValueError(f"Unknown init pose type for train: {init_pose_type}")


def _normalize_image(im, mean, std):
    # Bx3xHxW, 3x1x1
    return (im - mean) / std


def get_DIBR_models_renderer(cfg, data_ref, obj_names, output_db=True, gpu_id=None):
    """
    Args:
        output_db (bool): Compute and output image-space derivates of barycentrics.
    """
    model_dir = data_ref.model_dir
    obj_ids = [data_ref.obj2id[_obj] for _obj in obj_names]
    model_paths = [osp.join(model_dir, "obj_{:06d}.ply".format(obj_id)) for obj_id in obj_ids]

    texture_paths = None
    if data_ref.texture_paths is not None:
        texture_paths = [osp.join(model_dir, "obj_{:06d}.png".format(obj_id)) for obj_id in obj_ids]
    models = load_ply_models(
        model_paths=model_paths,
        texture_paths=texture_paths,
        vertex_scale=data_ref.vertex_scale,
        tex_resize=True,  # to reduce gpu memory usage
        width=512,
        height=512,
    )
    ren_DIBR = Renderer_DIBR(output_db=output_db, glctx_mode="manual", gpu_id=gpu_id)
    return models, ren_DIBR


def get_egl_renderer(cfg, data_ref, obj_names, gpu_id=None):
    from lib.egl_renderer.egl_renderer_v3 import EGLRenderer

    model_dir = data_ref.model_dir

    obj_ids = [data_ref.obj2id[_obj] for _obj in obj_names]
    model_paths = [osp.join(model_dir, "obj_{:06d}.ply".format(obj_id)) for obj_id in obj_ids]

    texture_paths = None
    if data_ref.texture_paths is not None:
        texture_paths = [osp.join(model_dir, "obj_{:06d}.png".format(obj_id)) for obj_id in obj_ids]

    ren = EGLRenderer(
        model_paths,
        texture_paths=texture_paths,
        vertex_scale=data_ref.vertex_scale,
        znear=data_ref.zNear,
        zfar=data_ref.zFar,
        K=data_ref.camera_matrix,  # may override later
        height=cfg.MODEL.DEEPIM.BACKBONE.INPUT_H,
        width=cfg.MODEL.DEEPIM.BACKBONE.INPUT_W,
        gpu_id=gpu_id,
        use_cache=True,
    )
    return ren


def get_out_coor(cfg, coor_x, coor_y, coor_z):
    if (coor_x.shape[1] == 1) and (coor_y.shape[1] == 1) and (coor_z.shape[1] == 1):
        coor_ = torch.cat([coor_x, coor_y, coor_z], dim=1)
    else:
        coor_ = torch.stack(
            [
                torch.argmax(coor_x, dim=1),
                torch.argmax(coor_y, dim=1),
                torch.argmax(coor_z, dim=1),
            ],
            dim=1,
        )
        # set the coordinats of background to (0, 0, 0)
        coor_[coor_ == cfg.MODEL.DEEPIM.XYZ_HEAD.XYZ_BIN] = 0
        # normalize the coordinates to [0, 1]
        coor_ = coor_ / float(cfg.MODEL.DEEPIM.XYZ_HEAD.XYZ_BIN - 1)

    return coor_


def get_out_mask(cfg, pred_mask):
    # (b,c,h,w)
    # output: (b, 1, h, w)
    mask_loss_type = cfg.MODEL.DEEPIM.MASK_HEAD.MASK_LOSS_TYPE
    bs, c, h, w = pred_mask.shape
    if mask_loss_type == "L1":
        assert c == 1, c
        mask_max = torch.max(pred_mask.view(bs, -1), dim=-1)[0].view(bs, 1, 1, 1)
        mask_min = torch.min(pred_mask.view(bs, -1), dim=-1)[0].view(bs, 1, 1, 1)
        # [0, 1]
        out_mask = (pred_mask - mask_min) / (mask_max - mask_min)  # + 1e-6)
    elif mask_loss_type in ["BCE", "RW_BCE", "dice"]:
        assert c == 1, c
        out_mask = torch.sigmoid(pred_mask)
    elif mask_loss_type == "CE":
        out_mask = torch.argmax(pred_mask, dim=1, keepdim=True)
    else:
        raise NotImplementedError(f"unknown mask loss type: {mask_loss_type}")
    return out_mask


def _zeros(_n, _c, _h, _w, dtype=torch.float32, device="cuda"):
    _tensor_kwargs = {"dtype": dtype, "device": device}
    return torch.zeros(_n, _c, _h, _w, **_tensor_kwargs).detach()


def _empty(_n, _c, _h, _w, dtype=torch.float32, device="cuda"):
    _tensor_kwargs = {"dtype": dtype, "device": device}
    return torch.empty(_n, _c, _h, _w, **_tensor_kwargs).detach()


def get_input_dim(cfg):
    backbone_cfg = cfg.MODEL.DEEPIM.BACKBONE
    if backbone_cfg.SHARED:
        dim = 6
        if backbone_cfg.INPUT_MASK:
            dim += 2
        if backbone_cfg.INPUT_DEPTH:
            dim += 2
        return dim
    else:
        dim_obs = 3
        dim_ren = 3
        if backbone_cfg.INPUT_MASK:
            dim_obs += 1
            dim_ren += 1
        if backbone_cfg.INPUT_DEPTH:
            dim_obs += 1
            dim_ren += 1
        return dim_obs, dim_ren


def boxes_to_masks(boxes, imH, imW, device="cuda", dtype=torch.float32):
    n_obj = boxes.shape[0]
    masks = _zeros(n_obj, 1, imH, imW, device=device, dtype=dtype)  # the square region of bbox
    for _i in range(n_obj):
        x1, y1, x2, y2 = boxes[_i]
        x1 = int(min(imW - 1, max(0, x1)))
        y1 = int(min(imH - 1, max(0, y1)))
        x2 = int(min(imW - 1, max(0, x2)))
        y2 = int(min(imH - 1, max(0, y2)))
        masks[_i, 0, y1 : y2 + 1, x1 : x2 + 1] = 1.0
    return masks


def vis_batch(cfg, batch, phase="train", refine_iter=1):
    from core.utils.utils import get_emb_show

    n_obj = batch["obj_cls"].shape[0]
    net_cfg = cfg.MODEL.DEEPIM
    backbone_cfg = net_cfg.BACKBONE
    loss_cfg = net_cfg.LOSS_CFG
    im_ids = batch["im_id"]
    # yapf: disable
    for i in range(n_obj):
        vis_dict = {f"img_{refine_iter}": (batch['img'][int(im_ids[i])].detach().cpu().numpy().transpose(1,2,0) * 255).astype('uint8')[:,:,::-1]}
        if loss_cfg.FLOW_LW > 0 and phase == 'train':
            vis_dict['zoom_depth_gl'] = batch['zoom_depth_gl'][i, 0].detach().cpu().numpy()
            vis_dict['zoom_depth_ren'] = batch['zoom_depth_ren'][i, 0].detach().cpu().numpy()
            vis_dict['zoom_depth_gl_ren_diff'] = vis_dict['zoom_depth_gl'] - vis_dict["zoom_depth_ren"]
            vis_dict['zoom_flow'] = flow2rgb(batch['zoom_flow'][i].detach().cpu().numpy().transpose(1,2,0))

        if loss_cfg.MASK_LW > 0 and phase == 'train':
            vis_dict['zoom_visib_mask'] = batch['zoom_visib_mask'][i].detach().cpu().numpy()
            vis_dict['zoom_trunc_mask'] = batch['zoom_trunc_mask'][i].detach().cpu().numpy()

        if "zoom_x" in batch:
            # ren data
            if net_cfg.INPUT_REN_TYPE == "rgb":
                vis_dict["zoom_img_ren"] = (batch['zoom_x'][i, :3].detach().cpu().numpy().transpose(1,2,0) * 255).astype('uint8')[:,:,::-1]
            elif net_cfg.INPUT_REN_TYPE == "xyz":
                vis_dict["zoom_xyz_ren"] = get_emb_show(batch["zoom_x"][i, :3].detach().cpu().numpy().transpose(1,2,0))
            else:
                raise ValueError(f"Invalid input_ren_type: {net_cfg.INPUT_REN_TYPE}")
            # obs data
            if net_cfg.INPUT_OBS_TYPE == "rgb":
                vis_dict['zoom_img_obs'] = (batch['zoom_x'][i, 3:6].detach().cpu().numpy().transpose(1,2,0) * 255).astype('uint8')[:,:,::-1]
            elif net_cfg.INPUT_OBS_TYPE == "xyz":
                vis_dict["zoom_xyz_obs"] = get_emb_show(batch["zoom_x"][i, 3:6].detach().cpu().numpy().transpose(1,2,0))
            else:
                raise ValueError(f"Invalid input_obs_type: {net_cfg.INPUT_OBS_TYPE}")

            if backbone_cfg.INPUT_MASK:
                vis_dict['zoom_mask_ren'] = batch['zoom_x'][i, 6].detach().cpu().numpy()
                vis_dict['zoom_mask_obs'] = batch['zoom_x'][i, 7].detach().cpu().numpy()
                if backbone_cfg.INPUT_DEPTH:
                    vis_dict['zoom_depth_ren'] = batch['zoom_x'][i, 8].detach().cpu().numpy()
                    vis_dict['zoom_depth_obs'] = batch['zoom_x'][i, 9].detach().cpu().numpy()
            else:  # no input mask
                if backbone_cfg.INPUT_DEPTH:
                    vis_dict['zoom_depth_ren'] = batch['zoom_x'][i, 6].detach().cpu().numpy()
                    vis_dict['zoom_depth_obs'] = batch['zoom_x'][i, 7].detach().cpu().numpy()
        else:  # unshared
            # ren data
            if net_cfg.INPUT_REN_TYPE == "rgb":
                vis_dict['zoom_img_ren'] = (batch['zoom_x_ren'][i, 0:3].detach().cpu().numpy().transpose(1,2,0) * 255).astype('uint8')[:,:,::-1]
            elif net_cfg.INPUT_REN_TYPE == "xyz":
                vis_dict['zoom_xyz_ren'] = get_emb_show(batch['zoom_x_ren'][i, 0:3].detach().cpu().numpy().transpose(1,2,0))
            else:
                raise ValueError(f"Invalid input_ren_type: {net_cfg.INPUT_REN_TYPE}")
            vis_dict['zoom_img_obs'] = (batch['zoom_x_obs'][i, 0:3].detach().cpu().numpy().transpose(1,2,0) * 255).astype('uint8')[:,:,::-1]
            if backbone_cfg.INPUT_MASK:
                vis_dict['zoom_mask_ren'] = batch['zoom_x_ren'][i, 3].detach().cpu().numpy()
                vis_dict['zoom_mask_obs'] = batch['zoom_x_obs'][i, 3].detach().cpu().numpy()
                if backbone_cfg.INPUT_DEPTH:
                    vis_dict['zoom_depth_ren'] = batch['zoom_x_ren'][i, 4].detach().cpu().numpy()
                    vis_dict['zoom_depth_obs'] = batch['zoom_x_obs'][i, 4].detach().cpu().numpy()
            else:  # no input mask
                if backbone_cfg.INPUT_DEPTH:
                    vis_dict['zoom_depth_ren'] = batch['zoom_x_ren'][i, 3].detach().cpu().numpy()
                    vis_dict['zoom_depth_obs'] = batch['zoom_x_obs'][i, 3].detach().cpu().numpy()

        if "zoom_depth_gl" in vis_dict and "zoom_visib_mask" in vis_dict:
            vis_dict["zoom_depth_gl_visib_mask_diff"] = heatmap((vis_dict["zoom_depth_gl"] > 0).astype("float32") - vis_dict["zoom_visib_mask"], to_rgb=True)
        show_titles = list(vis_dict.keys())
        show_ims = list(vis_dict.values())
        ncol = 4
        nrow = int(np.ceil(len(show_ims) / ncol))
        grid_show(show_ims, show_titles, row=nrow, col=ncol)
    # yapf: enable
