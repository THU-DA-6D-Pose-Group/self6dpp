import torch
import numpy as np
import math
import mmcv
from transforms3d.euler import euler2mat
from core.utils.pose_utils import euler2mat_torch


def aug_poses_normal(poses, std_rot=15, std_trans=[0.01, 0.01, 0.05], max_rot=45, min_z=0.1):
    """
    Args:
        poses (Tensor): [n,3,4]
        std_rot: deg, randomly chosen from cfg.INPUT.NOISE_ROT_STD_{TRAIN|TEST}, eg. (15, 10, 5, 2.5)
        std_trans: [dx, dy, dz], cfg.INPUT.NOISE_TRANS_STD_{TRAIN|TEST}
        max_rot: deg, cfg.INPUT.NOISE_ROT_MAX_{TRAIN|TEST}
        min_z: m, cfg.INPUT.INIT_TRANS_MIN_Z (z should not be smaller than some value)
    Returns:
        poses_aug: [n,3,4]
    """
    assert poses.ndim == 3, poses.shape
    poses_aug = poses.clone()
    bs = poses.shape[0]
    device = poses.device

    if isinstance(std_rot, (tuple, list)):
        std_rot = np.random.choice(std_rot)

    euler_noises_deg = torch.normal(mean=0, std=std_rot, size=(bs, 3)).to(device=device)
    if max_rot is not None:
        euler_noises_deg = euler_noises_deg.clamp(min=-max_rot, max=max_rot)

    rot_noises = euler2mat_torch(euler_noises_deg * math.pi / 180.0)  # (b,3,3)

    if mmcv.is_seq_of(std_trans, (tuple, list)) and isinstance(std_trans[0], (tuple, list)):  # list of tuple/list
        # randomly choose one setting
        sel_idx = np.random.choice(len(std_trans))
        sel_std_trans = std_trans[sel_idx]
    else:
        sel_std_trans = std_trans

    trans_noises = torch.normal(
        mean=torch.zeros_like(poses[:, :3, 3]),
        std=torch.tensor(sel_std_trans, device=device).view(1, 3),
    )
    poses_aug[:, :3, :3] = rot_noises @ poses[:, :3, :3]
    poses_aug[:, :3, 3] += trans_noises
    # z shoule >= min_z, or > 0
    min_z = max(min_z, 1e-4)
    poses_aug[:, 2, 3] = torch.clamp(poses_aug[:, 2, 3], min=min_z)
    return poses_aug


def aug_poses_normal_np(poses, std_rot=15, std_trans=[0.01, 0.01, 0.05], max_rot=45, min_z=0.1):
    """
    Args:
        poses (ndarray): [n,3,4]
        std_rot: deg, randomly chosen from cfg.INPUT.NOISE_ROT_STD_{TRAIN|TEST}
        std_trans: [dx, dy, dz], cfg.INPUT.NOISE_TRANS_STD_{TRAIN|TEST}
        max_rot: deg, cfg.INPUT.NOISE_ROT_MAX_{TRAIN|TEST}
    Returns:
        poses_aug (ndarray): [n,3,4]
    """
    assert poses.ndim == 3, poses.shape
    poses_aug = poses.copy()
    bs = poses.shape[0]

    if isinstance(std_rot, (tuple, list)):
        std_rot = np.random.choice(std_rot)

    euler_noises_deg = np.random.normal(loc=0, scale=std_rot, size=(bs, 3))
    if max_rot is not None:
        euler_noises_deg = np.clip(euler_noises_deg, -max_rot, max_rot)
    euler_noises_rad = euler_noises_deg * math.pi / 180.0
    rot_noises = np.array([euler2mat(*xyz) for xyz in euler_noises_rad])

    if mmcv.is_seq_of(std_trans, (tuple, list)) and isinstance(std_trans[0], (tuple, list)):
        sel_idx = np.random.choice(len(std_trans))
        sel_std_trans = std_trans[sel_idx]
    else:
        sel_std_trans = std_trans

    trans_noises = np.concatenate(
        [np.random.normal(loc=0, scale=std_trans_i, size=(bs, 1)) for std_trans_i in sel_std_trans],
        axis=1,
    )

    poses_aug[:, :3, :3] = rot_noises @ poses[:, :3, :3]
    poses_aug[:, :3, 3] += trans_noises
    # z should be >= min_z or > 0
    min_z = max(min_z, 1e-4)
    poses_aug[:, 2, 3] = np.clip(poses_aug[:, 2, 3], a_min=min_z, a_max=None)
    return poses_aug
