import numpy as np
import torch

from core.utils.utils import (
    allocentric_to_egocentric,
    allocentric_to_egocentric_torch,
    allo_to_ego_mat_torch,
)

from lib.pysixd import RT_transform
from core.utils.pose_utils import quat2mat_torch
from lib.utils.utils import dprint


def pose_from_delta_init(
    rot_deltas,
    trans_deltas,
    rot_inits,
    trans_inits,
    Ks=None,
    K_aware=False,
    delta_T_space="image",
    delta_T_weight=1.0,
    delta_z_style="cosypose",
    eps=1e-4,
    is_allo=False,
):
    """
    Args:
        rot_deltas: [b,3,3]
        trans_deltas: [b,3], vxvyvz, delta translations in image space
        rot_inits: [b,3,3]
        trans_inits: [b,3]
        Ks: if None, using constants 1
            otherwise use zoomed Ks
        K_aware: whether to use zoomed K
        delta_T_space: image | 3D
        delta_T_weight: deepim-pytorch uses 0.1, default 1.0
        delta_z_style: cosypose (_vz = ztgt / zsrc) | deepim (vz = log(zrsc/ztgt))
        eps:
        is_allo:
    Returns:
        rot_tgts, trans_tgts
    """
    bs = rot_deltas.shape[0]
    assert rot_deltas.shape == (bs, 3, 3)
    assert rot_inits.shape == (bs, 3, 3)
    assert trans_deltas.shape == (bs, 3)
    assert trans_inits.shape == (bs, 3)

    trans_deltas = trans_deltas * delta_T_weight  # TODO: maybe separate weight for x,y,z

    if delta_T_space == "image":
        # Translation in image space
        zsrc = trans_inits[:, [2]]  # [b,1]
        vz = trans_deltas[:, [2]]  # [b,1]
        if delta_z_style == "cosypose":
            # NOTE: directly predict vz = 1/exp(_vz)
            # log(zsrc/ztgt) = _vz ==> ztgt = 1/exp(_vz) * zsrc
            ztgt = vz * zsrc  # [b,1]
        else:  # deepim
            # vz = log(zsrc/ztgt)  ==> ztgt = zsrc / exp(vz)
            ztgt = torch.div(zsrc, torch.exp(vz))  # [b,1]

        if K_aware:
            assert Ks is not None and Ks.shape == (bs, 3, 3)
            vxvy = trans_deltas[:, :2]  # [b,2]
            fxfy = Ks[:, [0, 1], [0, 1]]  # [b,2]
        else:  # deepim: treat fx, fy as 1
            vxvy = trans_deltas[:, :2]  # [b,2]
            fxfy = torch.ones_like(vxvy)

        xy_src = trans_inits[:, :2]  # [b,2]
        xy_tgt = ztgt * (vxvy / fxfy + xy_src / zsrc)  # [b,2]
        trans_tgts = torch.cat([xy_tgt, ztgt], dim=-1)  # [b,3]
    elif delta_T_space == "3D":
        trans_tgts = trans_inits + trans_deltas
    else:
        raise ValueError("Unknown delta_T_space: {}".format(delta_T_space))

    if is_allo:
        ego_rot_deltas = allo_to_ego_mat_torch(trans_tgts, rot_deltas, eps=eps)
    else:
        ego_rot_deltas = rot_deltas

    # Rotation in camera frame
    rot_tgts = ego_rot_deltas @ rot_inits
    return rot_tgts, trans_tgts
