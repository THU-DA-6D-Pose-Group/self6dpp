import os.path as osp
import io
from functools import partial
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
import matplotlib.pyplot as plt
from einops import rearrange
from PIL import Image
from detectron2.layers import paste_masks_in_image
from fvcore.nn import smooth_l1_loss

from core.utils.camera_geometry import get_K_crop_resize
from core.utils.data_utils import xyz_to_region_batch, denormalize_image
from core.utils.utils import get_emb_show
from core.utils.edge_utils import compute_mask_edge_weights
from core.self6dpp.losses.mask_losses import weighted_ex_loss_probs, soft_dice_loss
from core.self6dpp.losses.depth_bp_chamfer_loss import depth_bp_chamfer_loss
from core.self6dpp.losses.pm_loss import PyPMLoss
from core.utils.zoom_utils import batch_crop_resize

from lib.torch_utils.color.lab import rgb_to_lab, normalize_lab
from lib.vis_utils.image import heatmap, grid_show


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    return img


def compute_self_loss(
    cfg,
    batch,
    pred_rot,
    pred_trans,
    pred_mask_prob,
    pred_full_mask_prob,
    pred_coor_x,
    pred_coor_y,
    pred_coor_z,
    pred_region,
    ren,
    ren_models,
    ssim_func=None,
    ms_ssim_func=None,
    perceptual_func=None,
    tb_writer=None,
    iteration=None,
):
    net_cfg = cfg.MODEL.POSE_NET
    self_loss_cfg = net_cfg.SELF_LOSS_CFG
    in_res = net_cfg.INPUT_RES
    out_res = net_cfg.OUTPUT_RES
    vis_i = 0
    vis_data = {}
    loss_dict = {}
    # for rendering data
    im_H, im_W = batch["gt_img"].shape[-2:]
    batch["K_renderer"] = batch["roi_cam"].clone()

    # get rendered mask/rgb/depth/xyz using DIBR
    cur_models = [ren_models[int(_l)] for _l in batch["roi_cls"]]

    if cfg.RENDERER.DIFF_RENDERER == "DIBR":
        ren_func = {"batch": partial(ren.render_batch), "batch_tex": partial(ren.render_batch_tex, uv_type="face")}[
            cfg.RENDERER.RENDER_TYPE
        ]
    else:
        raise ValueError("Unknown differentiable renderer type")

    ren_ret = ren_func(
        pred_rot,
        pred_trans,
        cur_models,
        Ks=batch["K_renderer"],
        width=im_W,
        height=im_H,
        mode=["color", "depth", "mask", "xyz", "prob"],
    )
    ren_img = rearrange(ren_ret["color"][..., [2, 1, 0]], "b h w c -> b c h w")  # bgr;[0,1]
    ren_mask = rearrange(ren_ret["mask"].to(torch.float32), "b h w -> b 1 h w")
    if "prob" in ren_ret.keys():
        ren_prob = rearrange(ren_ret["prob"].to(torch.float32), "b h w -> b 1 h w")
    else:
        ren_prob = ren_mask
    # ren_depth = rearrange(ren_ret["depth"], "b h w -> b 1 h w").contiguous()
    ren_depth = ren_ret["depth"]  # bhw
    ren_xyz = rearrange(ren_ret["xyz"], "b h w c -> b c h w")

    pseudo_mask = (batch["pseudo_mask_prob"] > 0.5).to(torch.float32)  # 64x64 roi level
    pseudo_mask_in_im = (batch["pseudo_mask_prob_in_im"] > 0.5).to(torch.float32)  # BHW

    if "pseudo_full_mask_prob" in batch.keys():
        pseudo_full_mask = (batch["pseudo_full_mask_prob"] > 0.5).to(torch.float32)  # 64x64 roi level
        pseudo_full_mask_in_im = (batch["pseudo_full_mask_prob_in_im"] > 0.5).to(torch.float32)  # BHW

    pseudo_mask_cal_ml = (
        pseudo_mask_in_im if cfg.MODEL.POSE_NET.SELF_LOSS_CFG.MASK_TYPE is "vis" else pseudo_full_mask_in_im
    )

    # compute edge weight (default lower edge weights because edge is not very accurate)
    if self_loss_cfg.MASK_WEIGHT_TYPE == "edge_lower":
        mask_weight_in_im = compute_mask_edge_weights(
            pseudo_mask_cal_ml[:, None, :, :], dilate_kernel_size=11, erode_kernel_size=11, edge_lower=True
        )
    elif self_loss_cfg.MASK_WEIGHT_TYPE == "edge_higher":
        mask_weight_in_im = compute_mask_edge_weights(
            pseudo_mask_cal_ml[:, None, :, :], dilate_kernel_size=11, erode_kernel_size=11, edge_lower=False
        )
    else:  # none (default)
        mask_weight_in_im = torch.ones_like(pseudo_mask_cal_ml[:, None, :, :])

    if tb_writer is not None:
        gt_img_vis = batch["gt_img"][vis_i].cpu().numpy()
        gt_img_vis = denormalize_image(gt_img_vis, cfg)[::-1].astype("uint8")
        vis_data["gt/image"] = rearrange(gt_img_vis, "c h w -> h w c")

        ren_img_vis = ren_img[vis_i].detach().cpu().numpy()
        ren_img_vis = denormalize_image(ren_img_vis, cfg)[::-1].astype("uint8")
        vis_data["ren/image"] = rearrange(ren_img_vis, "c h w -> h w c")

        pseudo_mask_vis = pseudo_mask[vis_i].detach().cpu().numpy()
        vis_data["pseudo/mask_roi"] = pseudo_mask_vis[0]

        pseudo_mask_prob_vis = batch["pseudo_mask_prob"][vis_i].cpu().numpy()
        vis_data["pseudo/vis_mask_prob_roi"] = pseudo_mask_prob_vis[0]

        if "pseudo_full_mask_prob" in batch.keys():
            pseudo_full_mask_vis = pseudo_full_mask[vis_i].detach().cpu().numpy()
            vis_data["pseudo/full_mask_roi"] = pseudo_full_mask_vis[0]

            pseudo_full_mask_prob_vis = batch["pseudo_full_mask_prob"][vis_i].cpu().numpy()
            vis_data["pseudo/vis_full_mask_prob_roi"] = pseudo_full_mask_prob_vis[0]

    # mask loss (init ren) -----------------------
    if self_loss_cfg.MASK_INIT_REN_LW > 0:
        if tb_writer is not None:
            ren_prob_roi = batch_crop_resize(
                ren_prob, batch["inst_rois"], out_H=pseudo_mask.shape[-2], out_W=pseudo_mask.shape[-1]
            )
            ren_prob_roi_vis = ren_prob_roi[vis_i].detach().cpu().numpy()
            vis_data["ren/prob_roi"] = ren_prob_roi_vis[0]

            mask_diff_pseudo_ren_vis = heatmap(pseudo_mask_vis[0] - ren_prob_roi_vis[0], to_rgb=True)
            vis_data["diff/mask_pseudo_ren"] = mask_diff_pseudo_ren_vis

        if self_loss_cfg.MASK_INIT_REN_LOSS_TYPE == "RW_BCE":
            loss_mask_init_ren = weighted_ex_loss_probs(
                ren_prob, pseudo_mask_cal_ml[:, None, :, :], weight=mask_weight_in_im
            )
        elif self_loss_cfg.MASK_INIT_REN_LOSS_TYPE == "dice":
            loss_mask_init_ren = soft_dice_loss(ren_prob, pseudo_mask_cal_ml[:, None, :, :], eps=0.002)
        else:
            raise NotImplementedError(
                "Not supported MASK_INIT_REN_LOSS_TYPE: {}".format(self_loss_cfg.MASK_INIT_REN_LOSS_TYPE)
            )
        loss_dict["loss_mask_init_ren"] = self_loss_cfg.MASK_INIT_REN_LW * loss_mask_init_ren

    if tb_writer is not None:
        pred_mask_prob_vis = pred_mask_prob[vis_i].detach().cpu().numpy()
        vis_data["pred/mask_prob_roi"] = pred_mask_prob_vis[0]
        vis_data["pred/mask_roi"] = (pred_mask_prob_vis > 0.5)[0].astype("float32")

    # mask loss (init pred) -----------------------
    if self_loss_cfg.MASK_INIT_PRED_LW > 0:
        if "vis" in self_loss_cfg.MASK_INIT_PRED_TYPE:
            loss_mask_init_pred = weighted_ex_loss_probs(
                pred_mask_prob, pseudo_mask, weight=torch.ones_like(pseudo_mask)
            )
            loss_dict["loss_mask_init_pred"] = self_loss_cfg.MASK_INIT_PRED_LW * loss_mask_init_pred
        if "full" in self_loss_cfg.MASK_INIT_PRED_TYPE:
            loss_full_mask_init_pred = weighted_ex_loss_probs(
                pred_full_mask_prob, pseudo_full_mask, weight=torch.ones_like(pseudo_full_mask)
            )
            loss_dict["loss_full_mask_init_pred"] = self_loss_cfg.MASK_INIT_PRED_LW * loss_full_mask_init_pred

    # prepare for color loss ----------------------
    if (
        self_loss_cfg.PERCEPT_LW > 0
        or self_loss_cfg.MS_SSIM_LW > 0
        or self_loss_cfg.LAB_LW > 0
        or self_loss_cfg.GEOM_LW > 0
    ):
        # crop/resize real and ren
        ren_img_roi = batch_crop_resize(ren_img, batch["inst_rois"], out_H=in_res, out_W=in_res)
        # ren_mask_roi = batch_crop_resize(ren_mask, batch["inst_rois"], out_H=256, out_W=256)
        gt_img_roi = batch["roi_gt_img"]
        # NOTE: use only visib parts to compute color loss
        pseudo_mask_roi = F.interpolate(pseudo_mask, size=(in_res, in_res), mode="nearest")
        if tb_writer is not None:
            roi_gt_img_vis = batch["roi_gt_img"][vis_i].cpu().numpy()
            roi_gt_img_vis = denormalize_image(roi_gt_img_vis, cfg)[::-1].astype("uint8")
            vis_data["gt/roi_image"] = rearrange(roi_gt_img_vis, "c h w -> h w c")

            gt_img_roi_masked_vis = (gt_img_roi[vis_i] * pseudo_mask_roi[vis_i]).cpu().numpy()
            gt_img_roi_masked_vis = denormalize_image(gt_img_roi_masked_vis, cfg)[::-1].astype("uint8")
            vis_data["gt/img_roi_masked"] = rearrange(gt_img_roi_masked_vis, "c h w -> h w c")

            ren_img_roi_vis = ren_img_roi[vis_i].detach().cpu().numpy()
            ren_img_roi_vis = denormalize_image(ren_img_roi_vis, cfg)[::-1].astype("uint8")
            vis_data["ren/img_roi"] = rearrange(ren_img_roi_vis, "c h w -> h w c")

            gt_ren_img_roi_vis = (0.5 * gt_img_roi[vis_i] + 0.5 * ren_img_roi[vis_i].detach()).cpu().numpy()
            gt_ren_img_roi_vis = denormalize_image(gt_ren_img_roi_vis, cfg)[::-1].astype("uint8")
            vis_data["diff/gt_ren_img_roi"] = rearrange(gt_ren_img_roi_vis, "c h w -> h w c")

    # perceptual loss on cropped region --------------------
    if self_loss_cfg.PERCEPT_LW > 0:
        assert perceptual_func is not None
        loss_percep_obj = perceptual_func(ren_img_roi, gt_img_roi * pseudo_mask_roi)
        loss_dict["loss_percep_obj"] = loss_percep_obj * self_loss_cfg.PERCEPT_LW

    # L1 loss in Lab space ---------------------------------------
    if self_loss_cfg.LAB_LW > 0:
        lab_gt_img_roi = rgb_to_lab(gt_img_roi[:, [2, 1, 0]])  # rgb, NCHW
        lab_gt_img_roi = normalize_lab(lab_gt_img_roi)  # [0, 1] NCHW
        # rgb, NHWC
        lab_ren_img_roi = rgb_to_lab(ren_img_roi[:, [2, 1, 0]].contiguous())
        lab_ren_img_roi = normalize_lab(lab_ren_img_roi)  # [0, 1], NCHW
        if self_loss_cfg.LAB_NO_L:
            loss_color_l1_obj = smooth_l1_loss(
                lab_gt_img_roi[:, 1:, :, :] * pseudo_mask_roi, lab_ren_img_roi[:, 1:, :, :], beta=0, reduction="sum"
            ) / max(1, pseudo_mask_roi.sum())
            loss_tag = "loss_color_ab_obj"
        else:
            loss_color_l1_obj = smooth_l1_loss(
                lab_gt_img_roi * pseudo_mask_roi, lab_ren_img_roi, beta=0, reduction="sum"
            ) / max(1, pseudo_mask_roi.sum())
            loss_tag = "loss_color_lab_obj"
        loss_dict[loss_tag] = self_loss_cfg.LAB_LW * loss_color_l1_obj

    # ms ssim loss ---------------------------------------------
    if self_loss_cfg.MS_SSIM_LW > 0:
        assert ms_ssim_func is not None
        loss_ms_ssim_obj = (1 - ms_ssim_func(gt_img_roi * pseudo_mask_roi, ren_img_roi)).mean()
        loss_dict["loss_ms_ssim"] = loss_ms_ssim_obj * self_loss_cfg.MS_SSIM_LW

    # depth chamfer loss --------------------------------------
    if self_loss_cfg.GEOM_LW > 0:
        # ren_depth_roi = batch_crop_resize(
        #     rearrange(ren_depth, "b h w -> b 1 h w"), batch["inst_rois"], out_H=in_res, out_W=in_res
        # )[:, 0]
        # gt_depth_roi = batch_crop_resize(
        #     rearrange(batch["depth"], "b h w -> b 1 h w"), batch["inst_rois"], out_H=in_res, out_W=in_res
        # )
        # gt_depth_roi_masked = (gt_depth_roi * pseudo_mask_roi)[:, 0]
        gt_depth_masked = batch["depth"] * pseudo_mask_in_im
        if self_loss_cfg.GEOM_LOSS_TYPE == "chamfer":
            # NOTE: real depths should be masked by pseudo mask
            # loss_depth_chamfer, loss_chamfer_center = depth_bp_chamfer_loss(
            #     ren_depth_roi,
            #     gt_depth_roi_masked,
            #     batch["roi_zoom_K_in"],
            #     distance_threshold=self_loss_cfg.CHAMFER_DIST_THR,
            #     center_lw=self_loss_cfg.CHAMFER_CENTER_LW,
            # )
            loss_depth_chamfer, loss_chamfer_center = depth_bp_chamfer_loss(
                ren_depth,
                gt_depth_masked,
                batch["roi_cam"],
                distance_threshold=self_loss_cfg.CHAMFER_DIST_THR,
                center_lw=self_loss_cfg.CHAMFER_CENTER_LW,
            )
            loss_dict["loss_chamfer"] = self_loss_cfg.GEOM_LW * loss_depth_chamfer
            if self_loss_cfg.CHAMFER_CENTER_LW > 0:
                loss_dict["loss_chamfer_center"] = loss_chamfer_center
        else:
            raise NotImplementedError("Unknown geom loss type: {}".format(self_loss_cfg.GEOM_LOSS_TYPE))

        if tb_writer is not None:
            pseudo_mask_in_im_vis = pseudo_mask_in_im[vis_i].detach().cpu().numpy()
            vis_data["pseudo/mask_in_im"] = pseudo_mask_in_im_vis

            gt_depth_vis = batch["depth"][vis_i].detach().cpu().numpy()
            vis_data["gt/depth"] = heatmap(gt_depth_vis, to_rgb=True)

            gt_depth_masked_vis = gt_depth_masked[vis_i].detach().cpu().numpy()
            vis_data["gt/depth_masked"] = heatmap(gt_depth_masked_vis, to_rgb=True)

            ren_depth_vis = ren_depth[vis_i].detach().cpu().numpy()
            vis_data["ren/depth"] = heatmap(ren_depth_vis, to_rgb=True)
            # ---------------------------------------------------------------------
            # gt_depth_roi_vis = gt_depth_roi[vis_i, 0].detach().cpu().numpy()
            # vis_data["gt/depth_roi"] = heatmap(gt_depth_roi_vis, to_rgb=True)

            # gt_depth_roi_masked_vis = gt_depth_roi_masked[vis_i].detach().cpu().numpy()
            # vis_data["gt/depth_roi_masked"] = heatmap(gt_depth_roi_masked_vis, to_rgb=True)

            # ren_depth_roi_vis = ren_depth_roi[vis_i].detach().cpu().numpy()
            # vis_data["ren/depth_roi"] = heatmap(ren_depth_roi_vis, to_rgb=True)

    # xyz init pred loss ---------------------------------
    if self_loss_cfg.XYZ_INIT_PRED_LW > 0:
        if self_loss_cfg.XYZ_INIT_PRED_LOSS_TYPE is "smoothL1":
            loss_init_pred_x = smooth_l1_loss(
                pred_coor_x * pseudo_mask, batch["pseudo_coor_x"] * pseudo_mask, beta=0, reduction="sum"
            ) / max(1, pseudo_mask.sum())
            loss_init_pred_y = smooth_l1_loss(
                pred_coor_y * pseudo_mask, batch["pseudo_coor_y"] * pseudo_mask, beta=0, reduction="sum"
            ) / max(1, pseudo_mask.sum())
            loss_init_pred_z = smooth_l1_loss(
                pred_coor_z * pseudo_mask, batch["pseudo_coor_z"] * pseudo_mask, beta=0, reduction="sum"
            ) / max(1, pseudo_mask.sum())
        elif self_loss_cfg.XYZ_INIT_PRED_LOSS_TYPE is "L1":
            loss_init_pred_x = nn.L1Loss(reduction="mean")(
                pred_coor_x * pseudo_mask, batch["pseudo_coor_x"] * pseudo_mask
            )
            loss_init_pred_y = nn.L1Loss(reduction="mean")(
                pred_coor_y * pseudo_mask, batch["pseudo_coor_y"] * pseudo_mask
            )
            loss_init_pred_z = nn.L1Loss(reduction="mean")(
                pred_coor_z * pseudo_mask, batch["pseudo_coor_z"] * pseudo_mask
            )
        else:
            raise ValueError("Unknow xyz init_pred loss type")

        loss_dict["loss_init_pred_x"] = loss_init_pred_x * self_loss_cfg.XYZ_INIT_PRED_LW
        loss_dict["loss_init_pred_y"] = loss_init_pred_y * self_loss_cfg.XYZ_INIT_PRED_LW
        loss_dict["loss_init_pred_z"] = loss_init_pred_z * self_loss_cfg.XYZ_INIT_PRED_LW

    # region loss -------------------------------
    if self_loss_cfg.REGION_INIT_PRED_LW > 0:
        loss_region_init_pred = nn.L1Loss(reduction="mean")(
            pred_region * pseudo_mask, batch["pseudo_region"] * pseudo_mask
        )
        loss_dict["loss_region_init_pred"] = loss_region_init_pred * self_loss_cfg.REGION_INIT_PRED_LW

    # self pm loss --------------------------------
    if self_loss_cfg.SELF_PM_CFG.loss_weight > 0:
        pm_loss_func = PyPMLoss(**self_loss_cfg.SELF_PM_CFG)
        loss_pm_dict = pm_loss_func(
            pred_rots=pred_rot,
            gt_rots=batch["pseudo_rot"],
            points=batch["roi_points"],
            pred_transes=pred_trans,
            gt_transes=batch["pseudo_trans"],
            extents=batch["roi_extent"],
            sym_infos=batch["sym_info"],
        )
        loss_dict.update(loss_pm_dict)

    # grid_show and write to tensorboard -----------------------------------------------------
    if tb_writer is not None:
        # TODO: maybe move this into self_engine and merge with other vis_data
        show_ims = list(vis_data.values())
        show_titles = list(vis_data.keys())
        n_per_col = 5
        nrow = int(np.ceil(len(show_ims) / n_per_col))
        fig = grid_show(show_ims, show_titles, row=nrow, col=n_per_col, show=False)
        im_grid = fig2img(fig)
        plt.close(fig)
        im_grid = torchvision.transforms.ToTensor()(im_grid)
        tb_writer.add_image("vis_im_grid", im_grid, global_step=iteration)
    return loss_dict


def batch_data_self(cfg, data, model_teacher=None, device="cuda", phase="train"):
    if phase != "train":
        return batch_data_test_self(cfg, data, device=device)

    # batch self training data
    net_cfg = cfg.MODEL.POSE_NET
    out_res = net_cfg.OUTPUT_RES

    tensor_kwargs = {"dtype": torch.float32, "device": device}
    assert model_teacher is not None
    assert not model_teacher.training, "teacher model must be in eval mode!"
    batch = {}
    # the image, infomation data and data from detection
    # augmented roi_image
    batch["roi_img"] = torch.stack([d["roi_img"] for d in data], dim=0).to(device, non_blocking=True)
    # original roi_image
    batch["roi_gt_img"] = torch.stack([d["roi_gt_img"] for d in data], dim=0).to(device, non_blocking=True)
    # original image
    batch["gt_img"] = torch.stack([d["gt_img"] for d in data], dim=0).to(device, non_blocking=True)
    im_H, im_W = batch["gt_img"].shape[-2:]

    if "depth" in data[0]:
        batch["depth"] = torch.stack([d["depth"] for d in data], dim=0).to(device, non_blocking=True)

    batch["roi_cls"] = torch.tensor([d["roi_cls"] for d in data], dtype=torch.long).to(device, non_blocking=True)
    bs = batch["roi_cls"].shape[0]
    if "roi_coord_2d" in data[0]:
        batch["roi_coord_2d"] = torch.stack([d["roi_coord_2d"] for d in data], dim=0).to(
            device=device, non_blocking=True
        )

    batch["roi_cam"] = torch.stack([d["cam"] for d in data], dim=0).to(device, non_blocking=True)
    batch["roi_center"] = torch.stack([d["bbox_center"] for d in data], dim=0).to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    batch["roi_scale"] = torch.tensor([d["scale"] for d in data], device=device, dtype=torch.float32)
    # for crop and resize
    rois_xy0 = batch["roi_center"] - batch["roi_scale"].view(bs, -1) / 2  # bx2
    rois_xy1 = batch["roi_center"] + batch["roi_scale"].view(bs, -1) / 2  # bx2
    batch["inst_rois"] = torch.cat([torch.arange(bs, **tensor_kwargs).view(-1, 1), rois_xy0, rois_xy1], dim=1)

    # for depth backprojection in input roi level
    in_res = net_cfg.INPUT_RES
    roi_resize_ratio_batch_in = in_res / batch["roi_scale"].view(bs, -1)
    batch["roi_zoom_K_in"] = get_K_crop_resize(batch["roi_cam"], rois_xy0, roi_resize_ratio_batch_in)

    batch["roi_wh"] = torch.stack([d["roi_wh"] for d in data], dim=0).to(device, non_blocking=True)
    batch["resize_ratio"] = torch.tensor([d["resize_ratio"] for d in data]).to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    batch["roi_extent"] = torch.stack([d["roi_extent"] for d in data], dim=0).to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    if "sym_info" in data[0]:
        batch["sym_info"] = [d["sym_info"] for d in data]

    if "roi_points" in data[0]:
        batch["roi_points"] = torch.stack([d["roi_points"] for d in data], dim=0).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
    if "roi_fps_points" in data[0]:
        batch["roi_fps_points"] = torch.stack([d["roi_fps_points"] for d in data], dim=0).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
    # get pose related pseudo labels from teacher model --------------------------
    with torch.no_grad():
        out_dict = model_teacher(
            batch["roi_img"],
            roi_classes=batch["roi_cls"],
            roi_cams=batch["roi_cam"],
            roi_whs=batch["roi_wh"],
            roi_centers=batch["roi_center"],
            resize_ratios=batch["resize_ratio"],
            roi_coord_2d=batch.get("roi_coord_2d", None),
            roi_coord_2d_rel=batch.get("roi_coord_2d_rel", None),
            roi_extents=batch.get("roi_extent", None),
            do_self=True,
        )
        # rot, trans, mask, coor_x, coor_y, coor_z
    if cfg.MODEL.PSEUDO_POSE_TYPE == "pose_refine" and "pose_refine" in data[0]:
        batch["pseudo_rot"] = torch.stack([d["pose_refine"][:3, :3] for d in data], dim=0).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        batch["pseudo_trans"] = torch.stack([d["pose_refine"][:3, 3] for d in data], dim=0).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
    elif cfg.MODEL.PSEUDO_POSE_TYPE == "pose_est" and "pose_est" in data[0]:
        batch["pseudo_rot"] = torch.stack([d["pose_est"][:3, :3] for d in data], dim=0).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        batch["pseudo_trans"] = torch.stack([d["pose_est"][:3, 3] for d in data], dim=0).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
    else:  # default use pose_init (online inferred by teacher)
        batch["pseudo_rot"] = out_dict["rot"]
        batch["pseudo_trans"] = out_dict["trans"]
    # batch["pseudo_mask"] = out_dict["mask"]
    batch["pseudo_mask_prob"] = mask_prob = out_dict["mask_prob"]
    # set threshold < 0: uint8 [0,255]
    pseudo_mask_prob_in_im = paste_masks_in_image(
        mask_prob[:, 0, :, :], batch["inst_rois"][:, 1:5], image_shape=(im_H, im_W), threshold=-1
    )
    batch["pseudo_mask_prob_in_im"] = pseudo_mask_prob_in_im.to(torch.float32) / 255

    if "full_mask_prob" in out_dict.keys():
        batch["pseudo_full_mask_prob"] = full_mask_prob = out_dict["full_mask_prob"]
        pseudo_full_mask_prob_in_im = paste_masks_in_image(
            full_mask_prob[:, 0, :, :], batch["inst_rois"][:, 1:5], image_shape=(im_H, im_W), threshold=-1
        )
        batch["pseudo_full_mask_prob_in_im"] = pseudo_full_mask_prob_in_im.to(torch.float32) / 255

    batch["pseudo_coor_x"] = coor_x = out_dict["coor_x"]
    batch["pseudo_coor_y"] = coor_y = out_dict["coor_y"]
    batch["pseudo_coor_z"] = coor_z = out_dict["coor_z"]
    # pseudo_coor = torch.cat([coor_x, coor_y, coor_z], dim=1)
    # batch["pseudo_region"] = xyz_to_region_batch(
    #     rearrange(pseudo_coor, "b c h w -> b h w c"), batch["roi_fps_points"], (mask_prob > 0.5).to(torch.float32)
    # )
    batch["pseudo_region"] = out_dict["region"]

    return batch


def batch_data_train_online_self(cfg, data, renderer, device="cuda"):
    # batch training data, rendering xyz online
    net_cfg = cfg.MODEL.POSE_NET
    g_head_cfg = net_cfg.GEO_HEAD
    batch = {}
    batch["roi_img"] = torch.stack([d["roi_img"] for d in data], dim=0).to(device, non_blocking=True)
    batch["roi_cls"] = torch.tensor([d["roi_cls"] for d in data], dtype=torch.long).to(device, non_blocking=True)
    bs = batch["roi_cls"].shape[0]
    if "roi_coord_2d" in data[0]:
        batch["roi_coord_2d"] = torch.stack([d["roi_coord_2d"] for d in data], dim=0).to(
            device=device, non_blocking=True
        )

    batch["roi_cam"] = torch.stack([d["cam"] for d in data], dim=0).to(device, non_blocking=True)
    batch["roi_center"] = torch.stack([d["bbox_center"] for d in data], dim=0).to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    batch["roi_scale"] = torch.tensor([d["scale"] for d in data], device=device, dtype=torch.float32)
    batch["resize_ratio"] = torch.tensor(
        [d["resize_ratio"] for d in data], device=device, dtype=torch.float32
    )  # out_res/scale
    # get crop&resized K -------------------------------------------
    roi_crop_xy_batch = batch["roi_center"] - batch["roi_scale"].view(bs, -1) / 2
    out_res = net_cfg.OUTPUT_RES
    roi_resize_ratio_batch = out_res / batch["roi_scale"].view(bs, -1)
    batch["roi_zoom_K"] = get_K_crop_resize(batch["roi_cam"], roi_crop_xy_batch, roi_resize_ratio_batch)
    # --------------------------------------------------------------
    batch["roi_wh"] = torch.stack([d["roi_wh"] for d in data], dim=0).to(device, non_blocking=True)
    batch["roi_extent"] = torch.stack([d["roi_extent"] for d in data], dim=0).to(
        device=device, dtype=torch.float32, non_blocking=True
    )  # [b,3]

    batch["roi_trans_ratio"] = torch.stack([d["trans_ratio"] for d in data], dim=0).to(device, non_blocking=True)
    # yapf: disable
    for key in [
        "roi_mask_trunc", "roi_mask_visib",
        "ego_rot", "trans",
        "roi_points",
    ]:
        if key in data[0]:
            dtype = torch.float32
            batch[key] = torch.stack([d[key] for d in data], dim=0).to(
                device=device, dtype=dtype, non_blocking=True
            )
    # yapf: enable
    if "sym_info" in data[0]:
        batch["sym_info"] = [d["sym_info"] for d in data]

    # rendering online xyz -----------------------------
    pc_obj_tensor = torch.cuda.FloatTensor(out_res, out_res, 4, device=device).detach()  # xyz
    roi_xyz_batch = torch.empty(bs, out_res, out_res, 3, dtype=torch.float32, device=device)
    for _i in range(bs):
        pose = np.hstack(
            [batch["ego_rot"][_i].detach().cpu().numpy(), batch["trans"][_i].detach().cpu().numpy().reshape(3, 1)]
        )
        renderer.render(
            [int(batch["roi_cls"][_i])],
            [pose],
            K=batch["roi_zoom_K"][_i].detach().cpu().numpy(),
            pc_obj_tensor=pc_obj_tensor,
        )
        roi_xyz_batch[_i].copy_(pc_obj_tensor[:, :, :3], non_blocking=True)
    # [bs, out_res, out_res]
    batch["roi_mask_obj"] = (
        (roi_xyz_batch[..., 0] != 0) & (roi_xyz_batch[..., 1] != 0) & (roi_xyz_batch[..., 2] != 0)
    ).to(torch.float32)
    batch["roi_mask_trunc"] = batch["roi_mask_trunc"] * batch["roi_mask_obj"]
    batch["roi_mask_visib"] = batch["roi_mask_visib"] * batch["roi_mask_obj"]

    if g_head_cfg.NUM_REGIONS > 1:  # get roi_region ------------------------
        batch["roi_fps_points"] = torch.stack([d["roi_fps_points"] for d in data], dim=0).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        batch["roi_region"] = xyz_to_region_batch(roi_xyz_batch, batch["roi_fps_points"], mask=batch["roi_mask_obj"])
    # normalize to [0, 1]
    batch["roi_xyz"] = rearrange(roi_xyz_batch, "b h w c -> b c h w") / batch["roi_extent"].view(bs, 3, 1, 1) + 0.5

    # get xyz bin if needed ---------------------------------
    loss_cfg = net_cfg.LOSS_CFG
    xyz_loss_type = loss_cfg.XYZ_LOSS_TYPE
    if ("CE" in xyz_loss_type) or ("cls" in net_cfg.NAME):
        # coordinates: [0, 1] to discrete [0, XYZ_BIN-1]
        roi_xyz_bin_batch = (
            (batch["roi_xyz"] * (g_head_cfg.XYZ_BIN - 1) + 0.5).clamp(min=0, max=g_head_cfg.XYZ_BIN).to(torch.long)
        )
        # set bg to XYZ_BIN
        roi_masks = {"trunc": batch["roi_mask_trunc"], "visib": batch["roi_mask_visib"], "obj": batch["roi_mask_obj"]}
        roi_mask_xyz = roi_masks[loss_cfg.XYZ_LOSS_MASK_GT]
        for _c in range(roi_xyz_bin_batch.shape[1]):
            roi_xyz_bin_batch[:, _c][roi_mask_xyz == 0] = g_head_cfg.XYZ_BIN
        batch["roi_xyz_bin"] = roi_xyz_bin_batch

    if cfg.TRAIN.VIS:
        vis_batch_self(cfg, batch, phase="train")
    return batch


def batch_data_test_self(cfg, data, device="cuda"):
    batch = {}

    # yapf: disable
    roi_keys = ["im_H", "im_W",
                "roi_img", "inst_id", "roi_coord_2d", "roi_cls", "score", "roi_extent",
                "bbox", "bbox_est", "bbox_mode", "roi_wh",
                "scale", "resize_ratio",
                ]
    for key in roi_keys:
        if key in ["roi_cls"]:
            dtype = torch.long
        else:
            dtype = torch.float32
        if key in data[0]:
            batch[key] = torch.cat([d[key] for d in data], dim=0).to(device=device, dtype=dtype, non_blocking=True)
    # yapf: enable

    batch["roi_cam"] = torch.cat([d["cam"] for d in data], dim=0).to(device, non_blocking=True)
    batch["roi_center"] = torch.cat([d["bbox_center"] for d in data], dim=0).to(device, non_blocking=True)
    for key in ["scene_im_id", "file_name", "model_info"]:
        # flatten the lists
        if key in data[0]:
            batch[key] = list(itertools.chain(*[d[key] for d in data]))

    return batch


def get_dibr_models_renderer(cfg, data_ref, obj_names, output_db=True, gpu_id=None):
    """
    Args:
        output_db (bool): Compute and output image-space derivates of barycentrics.
    """
    from lib.dr_utils.dib_renderer_x.renderer_dibr import load_ply_models, Renderer_dibr

    obj_ids = [data_ref.obj2id[_obj] for _obj in obj_names]

    model_scaled_root = data_ref.model_scaled_simple_dir
    obj_paths = [osp.join(model_scaled_root, "{}/textured.obj".format(_obj)) for _obj in obj_names]

    texture_paths = None
    if data_ref.texture_paths is not None:
        texture_paths = [osp.join(model_scaled_root, "{}/texture_map.png".format(_obj)) for _obj in obj_names]

    models = load_ply_models(
        obj_paths=obj_paths,
        texture_paths=texture_paths,
        vertex_scale=data_ref.vertex_scale,
        tex_resize=True,  # to reduce gpu memory usage
        width=512,
        height=512,
    )
    ren_dibr = Renderer_dibr(
        height=cfg.RENDERER.DIBR.HEIGHT, width=cfg.RENDERER.DIBR.WIDTH, mode=cfg.RENDERER.DIBR.MODE
    )
    return models, ren_dibr


def get_egl_renderer_self(cfg, data_ref, obj_names, gpu_id=None):
    """for rendering the targets (xyz) online."""
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
        height=cfg.MODEL.POSE_NET.OUTPUT_RES,
        width=cfg.MODEL.POSE_NET.OUTPUT_RES,
        gpu_id=gpu_id,
        use_cache=True,
    )
    return ren


def get_out_coor(cfg, coor_x, coor_y, coor_z):
    if (coor_x.shape[1] == 1) and (coor_y.shape[1] == 1) and (coor_z.shape[1] == 1):
        coor_ = torch.cat([coor_x, coor_y, coor_z], dim=1)
    else:
        coor_ = torch.stack(
            [torch.argmax(coor_x, dim=1), torch.argmax(coor_y, dim=1), torch.argmax(coor_z, dim=1)], dim=1
        )
        # set the coordinats of background to (0, 0, 0)
        coor_[coor_ == cfg.MODEL.POSE_NET.GEO_HEAD.XYZ_BIN] = 0
        # normalize the coordinates to [0, 1]
        coor_ = coor_ / float(cfg.MODEL.POSE_NET.GEO_HEAD.XYZ_BIN - 1)

    return coor_


def get_out_mask(cfg, pred_mask):
    # (b,c,h,w)
    # output: (b, 1, h, w)
    mask_loss_type = cfg.MODEL.POSE_NET.GEO_HEAD.MASK_LOSS_TYPE
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


def vis_batch_self(cfg, batch, phase="train"):
    n_obj = batch["roi_cls"].shape[0]
    # yapf: disable
    for i in range(n_obj):
        vis_dict = {"roi_img": (batch['roi_img'][i].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype('uint8')[:,:,::-1]}
        if phase == 'train':
            vis_dict['roi_mask_trunc'] = batch['roi_mask_trunc'][i].detach().cpu().numpy()
            vis_dict['roi_mask_visib'] = batch['roi_mask_visib'][i].detach().cpu().numpy()
            vis_dict['roi_mask_obj'] = batch['roi_mask_obj'][i].detach().cpu().numpy()

            vis_dict['roi_xyz'] = get_emb_show(batch['roi_xyz'][i].detach().cpu().numpy().transpose(1, 2, 0))

        show_titles = list(vis_dict.keys())
        show_ims = list(vis_dict.values())
        ncol = 4
        nrow = int(np.ceil(len(show_ims) / ncol))
        grid_show(show_ims, show_titles, row=nrow, col=ncol)
    # yapf: enable
