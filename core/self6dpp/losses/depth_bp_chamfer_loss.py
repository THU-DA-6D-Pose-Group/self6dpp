import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.csrc.torch_nndistance import torch_nndistance as NND
from fvcore.nn import smooth_l1_loss
from lib.pysixd.misc import backproject_th
from lib.vis_utils.image import heatmap


def depth_bp_chamfer_loss(ren_depths, real_depths, Ks, distance_threshold=0.05, center_lw=0):
    """
    Args:
        ren_depths: BHW
        real_depths: BHW
            target points: depth(masked) => backproject (K)
    """
    # TODO: a better threshold
    # distance_threshold = 0.05  # 0.05 # 0.025
    bs = len(ren_depths)
    num_valid = 0
    loss = torch.tensor(0.0).to(ren_depths)
    loss_center = torch.tensor(0.0).to(ren_depths)
    for i in range(bs):
        if Ks.ndim == 2:
            K = Ks
        else:
            K = Ks[i]

        real_pc_bp = backproject_th(real_depths[i], K)
        # TODO: outlier removal
        real_points_bp = real_pc_bp[real_pc_bp[:, :, 2] > 0]  # Nx3

        rend_pc_bp = backproject_th(ren_depths[i], K)
        rend_points_bp = rend_pc_bp[rend_pc_bp[:, :, 2] > 0]  # Nx3

        dist1, dist2 = NND.nnd(real_points_bp[None], rend_points_bp[None])
        if distance_threshold > 0:
            dist1 = dist1[dist1 < distance_threshold]
            dist2 = dist2[dist2 < distance_threshold]

        # if False:
        #     cv2.imshow('depth', heatmap(depths[i].detach().cpu().numpy()))
        #     cv2.waitKey(1)
        #     print(torch.mean(dist1), torch.mean(dist2))
        cur_loss = torch.mean(dist1) + torch.mean(dist2)
        if torch.isnan(cur_loss):
            continue
        loss += cur_loss

        if center_lw > 0:
            cur_center_loss = smooth_l1_loss(
                torch.mean(real_points_bp, 0),
                torch.mean(rend_points_bp, 0),
                beta=0,
                reduction="mean",
            )
            loss_center += cur_center_loss * center_lw

        num_valid += 1
    return loss / max(num_valid, 1), loss_center / max(num_valid, 1)
