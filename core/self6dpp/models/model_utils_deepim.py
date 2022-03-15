import torch
import copy
import numpy as np
from lib.pysixd.pose_error import re, te
from core.utils.pose_utils import quat2mat_torch
from core.utils.rot_reps import rot6d_to_mat_batch
from core.utils import lie_algebra, quaternion_lf
from .net_factory import HEADS


def get_rot_dim(rot_type):
    if rot_type in ["allo_quat", "ego_quat"]:
        rot_dim = 4
    elif rot_type in [
        "allo_log_quat",
        "ego_log_quat",
        "allo_lie_vec",
        "ego_lie_vec",
    ]:
        rot_dim = 3
    elif rot_type in ["allo_rot6d", "ego_rot6d"]:
        rot_dim = 6
    else:
        raise ValueError(f"Unknown rot_type: {rot_type}")
    return rot_dim


def get_rot_mat(rot, rot_type):
    if rot_type in ["ego_quat", "allo_quat"]:
        rot_m = quat2mat_torch(rot)
    elif rot_type in ["ego_log_quat", "allo_log_quat"]:
        # from latentfusion (lf)
        rot_m = quat2mat_torch(quaternion_lf.qexp(rot))
    elif rot_type in ["ego_lie_vec", "allo_lie_vec"]:
        rot_m = lie_algebra.lie_vec_to_rot(rot)
    elif rot_type in ["ego_rot6d", "allo_rot6d"]:
        rot_m = rot6d_to_mat_batch(rot)
    else:
        raise ValueError(f"Wrong pred_rot type: {rot_type}")
    return rot_m


def get_mask_dim(mask_loss_type):
    if mask_loss_type in ["L1", "BCE", "RW_BCE", "dice"]:
        mask_out_dim = 1
    elif mask_loss_type in ["CE"]:
        mask_out_dim = 2
    else:
        raise NotImplementedError(f"unknown mask loss type: {mask_loss_type}")

    return mask_out_dim


def get_pose_head(cfg):
    net_cfg = cfg.MODEL.DEEPIM
    pose_head_cfg = net_cfg.POSE_HEAD
    params_lr_list = []
    rot_type = pose_head_cfg.ROT_TYPE

    rot_dim = get_rot_dim(rot_type)
    pose_num_classes = net_cfg.NUM_CLASSES if pose_head_cfg.CLASS_AWARE else 1

    pose_head_init_cfg = copy.deepcopy(pose_head_cfg.INIT_CFG)
    pose_head_type = pose_head_init_cfg.pop("type")

    pose_head_init_cfg.update(rot_dim=rot_dim, num_classes=pose_num_classes)
    pose_head = HEADS[pose_head_type](**pose_head_init_cfg)
    if pose_head_cfg.FREEZE:
        for param in pose_head.parameters():
            with torch.no_grad():
                param.requires_grad = False
    else:
        params_lr_list.append(
            {
                "params": filter(lambda p: p.requires_grad, pose_head.parameters()),
                "lr": float(cfg.SOLVER.BASE_LR) * pose_head_cfg.LR_MULT,
            }
        )
    return pose_head, params_lr_list


def get_mask_head(cfg, is_test=False):
    net_cfg = cfg.MODEL.DEEPIM
    mask_head_cfg = net_cfg.MASK_HEAD
    params_lr_list = []
    if mask_head_cfg.ENABLED:
        if is_test and not cfg.TEST.OUTPUT_MASK:
            mask_head = None
        else:
            mask_dim = get_mask_dim(net_cfg.LOSS_CFG.MASK_LOSS_TYPE)
            mask_num_classes = net_cfg.NUM_CLASSES if mask_head_cfg.CLASS_AWARE else 1

            mask_head_init_cfg = copy.deepcopy(mask_head_cfg.INIT_CFG)
            mask_head_type = mask_head_init_cfg.pop("type")

            mask_head_init_cfg.update(num_classes=mask_num_classes, out_dim=mask_dim)
            mask_head = HEADS[mask_head_type](**mask_head_init_cfg)

            if mask_head_cfg.FREEZE:
                for param in mask_head.parameters():
                    with torch.no_grad():
                        param.requires_grad = False
            else:
                params_lr_list.append(
                    {
                        "params": filter(lambda p: p.requires_grad, mask_head.parameters()),
                        "lr": float(cfg.SOLVER.BASE_LR) * mask_head_cfg.LR_MULT,
                    }
                )
    else:
        mask_head = None
    return mask_head, params_lr_list


def get_flow_head(cfg, is_test=False):
    net_cfg = cfg.MODEL.DEEPIM
    flow_head_cfg = net_cfg.FLOW_HEAD
    params_lr_list = []
    if flow_head_cfg.ENABLED:
        if is_test and not cfg.TEST.OUTPUT_FLOW:
            flow_head = None
        else:
            flow_num_classes = net_cfg.NUM_CLASSES if flow_head_cfg.CLASS_AWARE else 1

            flow_head_init_cfg = copy.deepcopy(flow_head_cfg.INIT_CFG)
            flow_head_type = flow_head_init_cfg.pop("type")
            flow_head_init_cfg.update(num_classes=flow_num_classes)
            flow_head = HEADS[flow_head_type](**flow_head_init_cfg)

            if flow_head_cfg.FREEZE:
                for param in flow_head.parameters():
                    with torch.no_grad():
                        param.requires_grad = False
            else:
                params_lr_list.append(
                    {
                        "params": filter(lambda p: p.requires_grad, flow_head.parameters()),
                        "lr": float(cfg.SOLVER.BASE_LR) * flow_head_cfg.LR_MULT,
                    }
                )
    else:
        flow_head = None
    return flow_head, params_lr_list


def get_mask_prob(pred_mask, mask_loss_type):
    # (b,c,h,w)
    # output: (b, 1, h, w)
    bs, c, h, w = pred_mask.shape
    if mask_loss_type == "L1":
        assert c == 1, c
        mask_max = torch.max(pred_mask.view(bs, -1), dim=-1)[0].view(bs, 1, 1, 1)
        mask_min = torch.min(pred_mask.view(bs, -1), dim=-1)[0].view(bs, 1, 1, 1)
        # [0, 1]
        mask_prob = (pred_mask - mask_min) / (mask_max - mask_min)  # + 1e-6)
    elif mask_loss_type in ["BCE", "RW_BCE", "dice"]:
        assert c == 1, c
        mask_prob = torch.sigmoid(pred_mask)
    elif mask_loss_type == "CE":
        mask_prob = torch.softmax(pred_mask, dim=1, keepdim=True)[:, 1:2, :, :]
    else:
        raise NotImplementedError(f"unknown mask loss type: {mask_loss_type}")
    return mask_prob


def compute_mean_re_te(pred_transes, pred_rots, gt_transes, gt_rots):
    pred_transes = pred_transes.detach().cpu().numpy()
    pred_rots = pred_rots.detach().cpu().numpy()
    gt_transes = gt_transes.detach().cpu().numpy()
    gt_rots = gt_rots.detach().cpu().numpy()

    bs = pred_rots.shape[0]
    R_errs = np.zeros((bs,), dtype=np.float32)
    T_errs = np.zeros((bs,), dtype=np.float32)
    for i in range(bs):
        R_errs[i] = re(pred_rots[i], gt_rots[i])
        T_errs[i] = te(pred_transes[i], gt_transes[i])
    return R_errs.mean(), T_errs.mean()
