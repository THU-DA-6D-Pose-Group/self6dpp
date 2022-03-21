# encoding: utf-8
"""This file includes necessary params, info."""
import os
import mmcv
import os.path as osp

import numpy as np

# ---------------------------------------------------------------- #
# ROOT PATH INFO
# ---------------------------------------------------------------- #
cur_dir = osp.abspath(osp.dirname(__file__))
root_dir = osp.normpath(osp.join(cur_dir, ".."))
output_dir = osp.join(root_dir, "output")  # directory storing experiment data (result, model checkpoints, etc).

data_root = osp.join(root_dir, "datasets")

# ---------------------------------------------------------------- #
# HB DATASET
# ---------------------------------------------------------------- #
dataset_root = osp.join(data_root, "hb_bench_driller_phone")
# NOTE: the gt provided is based on lm models
model_dir = osp.join(dataset_root, "models_lm")
model_eval_dir = osp.join(dataset_root, "models_lm_eval")
vertex_scale = 0.001
# scaled models (.obj)
model_scaled_simple_dir = osp.join(dataset_root, "models_lm_scaled_f5k")  # copied from lm

# object info
objects = ["benchvise", "driller", "phone"]
id2obj = {
    2: "benchvise",
    7: "driller",
    21: "phone",
}  # bop 2, hb-v1:1  # bop 7, hb-v1:6  # bop 21, hb-v1:20
obj_num = len(id2obj)
obj2id = {_name: _id for _id, _name in id2obj.items()}

model_paths = [osp.join(model_dir, "obj_{:06d}.ply").format(_id) for _id in id2obj]
texture_paths = None
model_colors = [((i + 1) * 10, (i + 1) * 10, (i + 1) * 10) for i in range(obj_num)]  # for renderer

# use diameters of lm models since the poses are using lm models
diameters = np.array([247.50624233, 261.47178102, 212.35825148]) / 1000.0
# diameters = np.array([257.407, 263.7182, 212.6029]) / 1000.0

# Camera info
width = 640
height = 480
zNear = 0.25
zFar = 6.0
center = (height / 2, width / 2)

# NOTE: this is the linemod one
camera_matrix = camera_matrix_lm = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])
# real one:
camera_matrix_hb = np.array([[537.4799, 0, 318.8965], [0, 536.1447, 238.3781], [0, 0, 1]])


def get_models_info():
    """key is str(obj_id)"""
    models_info_path = osp.join(model_dir, "models_info.json")
    assert osp.exists(models_info_path), models_info_path
    models_info = mmcv.load(models_info_path)  # key is str(obj_id)
    return models_info


def get_fps_points():
    """key is str(obj_id) generated by
    core/gdrn_modeling/tools/hb_bdp/hb_bdp_1_compute_fps.py."""
    fps_points_path = osp.join(model_dir, "fps_points.pkl")
    assert osp.exists(fps_points_path), fps_points_path
    fps_dict = mmcv.load(fps_points_path)
    return fps_dict


def get_keypoints_3d():
    """key is str(obj_id) generated by
    core/roi_pvnet/tools/hb_bdp/hb_bdp_1_compute_keypoints_3d.py."""
    keypoints_3d_path = osp.join(model_dir, "keypoints_3d.pkl")
    assert osp.exists(keypoints_3d_path), keypoints_3d_path
    kpts_dict = mmcv.load(keypoints_3d_path)
    return kpts_dict