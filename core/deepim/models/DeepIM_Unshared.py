import copy
import logging
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.utils.solver_utils import build_optimizer_with_params
from detectron2.utils.events import get_event_storage
from mmcv.runner import load_checkpoint

from ..losses.flow_loss import multiscaleEPE, one_scale_EPE

from ..losses.l2_loss import L2Loss
from ..losses.pm_loss import PyPMLoss
from ..losses.rot_loss import angular_distance, rot_l2_loss
from .model_utils import (
    compute_mean_re_te,
    get_mask_prob,
    get_flow_head,
    get_mask_head,
    get_pose_head,
    get_rot_mat,
)

from .pose_from_delta_init import pose_from_delta_init
from .net_factory import BACKBONES

logger = logging.getLogger(__name__)


class DeepIM_Unshared(nn.Module):
    def __init__(self, cfg, backbone, backbone_ren, pose_head=None, mask_head=None, flow_head=None, renderer=None):
        """use the unshared backbone for ren/obs."""
        super().__init__()
        assert cfg.MODEL.DEEPIM.NAME == "DeepIM_Unshared", cfg.MODEL.DEEPIM.NAME
        self.cfg = cfg
        self.renderer = renderer  # no need
        # backbones
        self.backbone = backbone
        self.backbone_ren = backbone_ren
        # heads
        self.pose_head = pose_head
        self.mask_head = mask_head
        self.flow_head = flow_head

        # uncertainty multi-task loss weighting
        # https://github.com/Hui-Li/multi-task-learning-example-PyTorch/blob/master/multi-task-learning-example-PyTorch.ipynb
        # a = log(sigma^2)
        # L*exp(-a) + a  or  L*exp(-a) + log(1+exp(a))
        # self.log_vars = nn.Parameter(torch.tensor([0, 0], requires_grad=True, dtype=torch.float32).cuda())
        if cfg.MODEL.DEEPIM.USE_MTL:
            # yapf: disable
            self.loss_names = [
                "mask",
                "flow", "flow4",
                "PM_R", "PM_xy", "PM_z", "PM_xy_noP", "PM_z_noP", "PM_T", "PM_T_noP",
                "z", "trans_xy", "trans_z",
                "rot"
            ]
            # yapf: enable
            for loss_name in self.loss_names:
                self.register_parameter(
                    f"log_var_{loss_name}",
                    nn.Parameter(torch.tensor([0.0], requires_grad=True, dtype=torch.float32)),
                )

    def forward(
        self,
        x,
        x_ren,
        init_pose,
        K_zoom,
        obj_class=None,
        gt_ego_rot=None,
        gt_trans=None,
        gt_flow=None,
        gt_mask_trunc=None,
        gt_mask_visib=None,
        gt_points=None,
        obj_extent=None,
        sym_info=None,
        do_loss=False,
        cur_iter=0,
        roi_coord_2d=None,
    ):
        """
        x: zoomed obs inputs
        x_ren: zoomed ren inputs
        """
        cfg = self.cfg
        net_cfg = cfg.MODEL.DEEPIM
        backbone_cfg = net_cfg.BACKBONE
        pose_head_cfg = net_cfg.POSE_HEAD
        mask_head_cfg = net_cfg.MASK_HEAD
        flow_head_cfg = net_cfg.FLOW_HEAD

        num_classes = net_cfg.NUM_CLASSES
        device = x.device
        bs = x.shape[0]

        # backbone forward
        conv_feat_obs = self.backbone(x)
        conv_feat_ren = self.backbone_ren(x_ren)
        if isinstance(conv_feat_obs, (tuple, list)) and len(conv_feat_obs) == 1:
            conv_feat_obs = conv_feat_obs[0]
        if isinstance(conv_feat_ren, (tuple, list)) and len(conv_feat_ren) == 1:
            conv_feat_ren = conv_feat_ren[0]
        # fuse obs ren conv feat
        conv_feat_obs_ren = torch.cat([conv_feat_obs, conv_feat_ren], dim=1)

        # predict mask obs ---------------------------------------------------
        if (self.mask_head is not None) and (do_loss or cfg.TEST.OUTPUT_MASK):
            mask = self.mask_head(conv_feat_obs)
            mask_h, mask_w = mask.shape[-2:]
            if mask_head_cfg.CLASS_AWARE:
                assert obj_class is not None
                mask = mask.view(bs, num_classes, self.mask_head.out_dim, mask_h, mask_w)
                mask = mask[torch.arange(bs).to(device), obj_class]
            mask_out = F.interpolate(mask, size=x.shape[-2:], mode="bilinear")
        else:
            mask_out = None

        # predict flow ---------------------------------------------------
        if (self.flow_head is not None) and (do_loss or cfg.TEST.OUTPUT_FLOW):
            flow = self.flow_head(conv_feat_obs_ren)  # assume stride 4
            flow_h, flow_w = flow.shape[-2:]
            if flow_head_cfg.CLASS_AWARE:
                assert obj_class is not None
                flow = flow.view(bs, num_classes, self.flow_head.out_dim, flow_h, flow_w)
                flow = flow[torch.arange(bs).to(device), obj_class]
        else:
            flow = None

        # predict delta R/T --------------------------------------------
        flat_conv_feat = conv_feat_obs_ren.flatten(2)
        if net_cfg.FLAT_OP == "flatten":
            flat_conv_feat = flat_conv_feat.flatten(1)
        elif net_cfg.FLAT_OP == "avg":
            flat_conv_feat = flat_conv_feat.mean(-1)  # spatial global average pooling
        elif net_cfg.FLAT_OP == "avg-max-min":
            flat_conv_feat = torch.cat(
                [
                    flat_conv_feat.mean(-1),
                    flat_conv_feat.max(-1)[0],
                    flat_conv_feat.min(-1)[0],
                ],
                dim=-1,
            )
        elif net_cfg.FLAT_OP == "avg-max":
            flat_conv_feat = torch.cat([flat_conv_feat.mean(-1), flat_conv_feat.max(-1)[0]], dim=-1)
        else:
            raise ValueError(f"Unknown FLAT_OP: {net_cfg.FLAT_OP}")

        rot_deltas_, trans_deltas_ = self.pose_head(flat_conv_feat)

        if pose_head_cfg.CLASS_AWARE:
            assert obj_class is not None
            rot_deltas_ = rot_deltas_.view(bs, num_classes, self.pose_head.rot_dim)
            rot_deltas_ = rot_deltas_[torch.arange(bs).to(device), obj_class]
            trans_deltas_ = trans_deltas_.view(bs, num_classes, 3)
            trans_deltas_ = trans_deltas_[torch.arange(bs).to(device), obj_class]

        # convert pred_rot to rot mat -------------------------
        rot_m_deltas = get_rot_mat(rot_deltas_, rot_type=pose_head_cfg.ROT_TYPE)
        # rot_m_deltas, trans_deltas, init_pose --> ego pose -----------------------------
        pred_ego_rot, pred_trans = pose_from_delta_init(
            rot_deltas=rot_m_deltas,
            trans_deltas=trans_deltas_,
            rot_inits=init_pose[:, :3, :3],
            trans_inits=init_pose[:, :3, 3],
            Ks=K_zoom,  # zoomed Ks
            K_aware=pose_head_cfg.T_TRANSFORM_K_AWARE,
            delta_T_space=pose_head_cfg.DELTA_T_SPACE,
            delta_T_weight=pose_head_cfg.DELTA_T_WEIGHT,
            delta_z_style=pose_head_cfg.DELTA_Z_STYLE,
            eps=1e-4,
            is_allo="allo" in pose_head_cfg.ROT_TYPE,
        )
        pred_pose = torch.cat([pred_ego_rot, pred_trans.view(-1, 3, 1)], dim=-1)
        out_dict = {f"pose_{cur_iter}": pred_pose}
        if not do_loss:  # test
            if cfg.TEST.OUTPUT_MASK:
                # TODO: flow output
                out_dict.update({"mask": get_mask_prob(mask_out, net_cfg.LOSS_CFG.MASK_LOSS_TYPE)})
        else:
            assert gt_ego_rot is not None and (gt_trans is not None)
            mean_re, mean_te = compute_mean_re_te(pred_trans, pred_ego_rot, gt_trans, gt_ego_rot)
            # yapf: disable
            vis_dict = {
                f"vis/error_R_{cur_iter}": mean_re,  # deg
                f"vis/error_t_{cur_iter}": mean_te * 100,  # cm
                f"vis/error_tx_{cur_iter}": np.abs(
                    pred_trans[0, 0].detach().item() - gt_trans[0, 0].detach().item()) * 100,  # cm
                f"vis/error_ty_{cur_iter}": np.abs(
                    pred_trans[0, 1].detach().item() - gt_trans[0, 1].detach().item()) * 100,  # cm
                f"vis/error_tz_{cur_iter}": np.abs(
                    pred_trans[0, 2].detach().item() - gt_trans[0, 2].detach().item()) * 100,  # cm
                f"vis/tx_pred_{cur_iter}": pred_trans[0, 0].detach().item(),
                f"vis/ty_pred_{cur_iter}": pred_trans[0, 1].detach().item(),
                f"vis/tz_pred_{cur_iter}": pred_trans[0, 2].detach().item(),
                f"vis/tx_delta_{cur_iter}": trans_deltas_[0, 0].detach().item(),
                f"vis/ty_delta_{cur_iter}": trans_deltas_[0, 1].detach().item(),
                f"vis/tz_delta_{cur_iter}": trans_deltas_[0, 2].detach().item(),
                f"vis/tx_gt_{cur_iter}": gt_trans[0, 0].detach().item(),
                f"vis/ty_gt_{cur_iter}": gt_trans[0, 1].detach().item(),
                f"vis/tz_gt_{cur_iter}": gt_trans[0, 2].detach().item(),
            }

            loss_dict = self.deepim_loss(
                out_rot=pred_ego_rot, gt_rot=gt_ego_rot,
                out_trans=pred_trans, gt_trans=gt_trans,
                gt_points=gt_points, sym_info=sym_info, extent=obj_extent,
                # mask loss
                out_mask=mask_out, gt_mask_trunc=gt_mask_trunc, gt_mask_visib=gt_mask_visib,
                # flow loss
                out_flow=flow, gt_flow=gt_flow,
            )

            if net_cfg.USE_MTL:
                for _name in self.loss_names:
                    if f"loss_{_name}" in loss_dict:
                        vis_dict[f"vis_lw/{_name}_{cur_iter}"] = torch.exp(
                            -getattr(self, f"log_var_{_name}")).detach().item()
            for _k, _v in vis_dict.items():
                if "vis/" in _k or "vis_lw/" in _k:
                    if isinstance(_v, torch.Tensor):
                        _v = _v.item()
                    vis_dict[_k] = _v
            # yapf: enable
            storage = get_event_storage()
            storage.put_scalars(**vis_dict)

            return out_dict, loss_dict
        return out_dict

    def deepim_loss(
        self,
        out_rot,
        out_trans,
        gt_rot=None,
        gt_trans=None,
        gt_points=None,
        sym_info=None,
        extent=None,
        out_mask=None,
        gt_mask_trunc=None,
        gt_mask_visib=None,
        gt_mask_obj=None,
        out_flow=None,
        gt_flow=None,
    ):
        cfg = self.cfg
        net_cfg = cfg.MODEL.DEEPIM
        pose_head_cfg = net_cfg.POSE_HEAD
        mask_head_cfg = net_cfg.MASK_HEAD
        loss_cfg = net_cfg.LOSS_CFG

        loss_dict = {}

        # point matching loss ---------------
        if loss_cfg.PM_LW > 0:
            assert (gt_points is not None) and (gt_trans is not None) and (gt_rot is not None)
            loss_func = PyPMLoss(
                loss_type=loss_cfg.PM_LOSS_TYPE,
                beta=loss_cfg.PM_SMOOTH_L1_BETA,
                reduction="mean",
                loss_weight=loss_cfg.PM_LW,
                norm_by_extent=loss_cfg.PM_NORM_BY_EXTENT,
                symmetric=loss_cfg.PM_LOSS_SYM,
                disentangle_t=loss_cfg.PM_DISENTANGLE_T,
                disentangle_z=loss_cfg.PM_DISENTANGLE_Z,
                t_loss_use_points=loss_cfg.PM_T_USE_POINTS,
                r_only=loss_cfg.PM_R_ONLY,
            )
            loss_pm_dict = loss_func(
                pred_rots=out_rot,
                gt_rots=gt_rot,
                points=gt_points,
                pred_transes=out_trans,
                gt_transes=gt_trans,
                extents=extent,
                sym_infos=sym_info,
            )
            loss_dict.update(loss_pm_dict)

        # rot_loss ----------
        if loss_cfg.ROT_LW > 0:
            if loss_cfg.ROT_LOSS_TYPE == "angular":
                loss_dict["loss_rot"] = angular_distance(out_rot, gt_rot)
            elif loss_cfg.ROT_LOSS_TYPE == "L2":
                loss_dict["loss_rot"] = rot_l2_loss(out_rot, gt_rot)
            else:
                raise ValueError(f"Unknown rot loss type: {loss_cfg.ROT_LOSS_TYPE}")
            loss_dict["loss_rot"] *= loss_cfg.ROT_LW

        gt_masks = {
            "trunc": gt_mask_trunc,
            "visib": gt_mask_visib,
            "obj": gt_mask_obj,
        }

        # mask loss ----------------------------------
        if (out_mask is not None) and (not mask_head_cfg.FREEZE) and (loss_cfg.MASK_LW > 0):
            mask_loss_type = loss_cfg.MASK_LOSS_TYPE
            gt_mask = gt_masks[loss_cfg.MASK_LOSS_GT]
            if mask_loss_type == "L1":
                loss_dict["loss_mask"] = nn.L1Loss(reduction="mean")(out_mask[:, 0, :, :], gt_mask)
            elif mask_loss_type == "BCE":
                loss_dict["loss_mask"] = nn.BCEWithLogitsLoss(reduction="mean")(out_mask[:, 0, :, :], gt_mask)
            elif mask_loss_type == "CE":
                loss_dict["loss_mask"] = nn.CrossEntropyLoss(reduction="mean")(out_mask, gt_mask.long())
            else:
                raise NotImplementedError(f"Unknown mask loss type: {mask_loss_type}")
            loss_dict["loss_mask"] *= loss_cfg.MASK_LW

        # flow loss
        if (out_flow is not None) and (loss_cfg.FLOW_LW > 0):
            assert gt_flow is not None
            if isinstance(out_flow, torch.Tensor):
                loss_flow4 = one_scale_EPE(out_flow, gt_flow, mean=True)
                loss_dict["loss_flow4"] = loss_flow4 * loss_cfg.FLOW_LW
            else:  # list
                loss_flow = multiscaleEPE(out_flow, gt_flow, mean=True)
                loss_dict["loss_flow"] = loss_flow * loss_cfg.FLOW_LW

        if net_cfg.USE_MTL:
            for _k in loss_dict:
                _name = _k.replace("loss_", "log_var_")
                cur_log_var = getattr(self, _name)
                loss_dict[_k] = loss_dict[_k] * torch.exp(-cur_log_var) + torch.log(1 + torch.exp(cur_log_var))
        return loss_dict


def build_model_optimizer(cfg, is_test=False):
    net_cfg = cfg.MODEL.DEEPIM
    backbone_cfg = net_cfg.BACKBONE
    backbone_ren_cfg = net_cfg.BACKBONE_REN
    assert (not backbone_cfg.SHARED) and backbone_ren_cfg.ENABLED, "The backbone for rendered input must be enabled!"

    params_lr_list = []

    # backbone obs ----------------------------------------------------------
    init_backbone_args = copy.deepcopy(backbone_cfg.INIT_CFG)
    backbone_type = init_backbone_args.pop("type")
    if "timm/" in backbone_type:
        init_backbone_args["model_name"] = backbone_type.split("/")[-1]

    backbone = BACKBONES[backbone_type](**init_backbone_args)
    if backbone_cfg.FREEZE:  # TODO: support freeze first few stages
        for param in backbone.parameters():
            with torch.no_grad():
                param.requires_grad = False
    else:
        params_lr_list.append(
            {
                "params": filter(lambda p: p.requires_grad, backbone.parameters()),
                "lr": float(cfg.SOLVER.BASE_LR),
            }
        )

    # backbone ren --------------------------------------------------------
    backbone_ren_type = backbone_ren_cfg.INIT_CFG.pop("type")
    init_backbone_ren_args = copy.deepcopy(backbone_ren_cfg.INIT_CFG)
    if "timm/" in backbone_ren_type:
        init_backbone_ren_args["model_name"] = backbone_ren_type.split("/")[-1]

    backbone_ren = BACKBONES[backbone_ren_type](**init_backbone_ren_args)
    if backbone_ren_cfg.FREEZE:
        for param in backbone_ren.parameters():
            with torch.no_grad():
                param.requires_grad = False
    else:
        params_lr_list.append(
            {
                "params": filter(lambda p: p.requires_grad, backbone_ren.parameters()),
                "lr": float(cfg.SOLVER.BASE_LR),
            }
        )

    # pose head -----------------------------------------------------
    pose_head, pose_head_params = get_pose_head(cfg)
    params_lr_list.extend(pose_head_params)

    # mask head ----------------------------------------------------
    mask_head, mask_head_params = get_mask_head(cfg, is_test=is_test)
    params_lr_list.extend(mask_head_params)

    # flow head (do not use this for flownet)------------------------------------------
    flow_head, flow_head_params = get_flow_head(cfg, is_test=is_test)
    params_lr_list.extend(flow_head_params)

    # ================================================
    # build model
    model = DeepIM_Unshared(
        cfg,
        backbone,
        backbone_ren,
        pose_head=pose_head,
        mask_head=mask_head,
        flow_head=flow_head,
    )
    if net_cfg.USE_MTL:
        params_lr_list.append(
            {
                "params": filter(
                    lambda p: p.requires_grad,
                    [_param for _name, _param in model.named_parameters() if "log_var" in _name],
                ),
                "lr": float(cfg.SOLVER.BASE_LR),
            }
        )

    # get optimizer
    if is_test:
        optimizer = None
    else:
        optimizer = build_optimizer_with_params(cfg, params_lr_list)

    if cfg.MODEL.WEIGHTS == "":
        ## backbone initialization
        backbone_pretrained = backbone_cfg.get("PRETRAINED", "")
        if backbone_pretrained == "":
            logger.warning("Randomly initialize weights for backbone!")
        elif backbone_pretrained in ["timm", "internal"]:
            # skip if it has already been initialized by pretrained=True
            logger.info("Check if the backbone has been initialized with its own method!")
        else:
            # initialize backbone with official weights
            tic = time.time()
            logger.info(f"load backbone weights from: {backbone_pretrained}")
            load_checkpoint(
                model.backbone,
                backbone_pretrained,
                strict=False,
                logger=logger,
            )
            logger.info(f"load backbone weights took: {time.time() - tic}s")

        ## backbone ren initialization
        backbone_ren_pretrained = backbone_ren_cfg.get("PRETRAINED", "")
        if backbone_ren_pretrained == "":
            logger.warning("Randomly initialize weights for backbone ren!")
        elif backbone_ren_pretrained in ["timm", "internal"]:
            # skip if it has already been initialized by pretrained=True
            logger.info("Check if the backbone_ren has been initialized with its own method!")
        else:
            # initialize backbone_ren with official weights
            tic = time.time()
            logger.info(f"load backbone_ren weights from: {backbone_ren_pretrained}")
            load_checkpoint(
                model.backbone_ren,
                backbone_ren_pretrained,
                strict=False,
                logger=logger,
            )
            logger.info(f"load backbone_ren weights took: {time.time() - tic}s")

    model.to(torch.device(cfg.MODEL.DEVICE))
    return model, optimizer
