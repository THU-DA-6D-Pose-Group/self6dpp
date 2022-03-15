import mmcv
import numpy as np
import os.path as osp
import sys
import math
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from detectron2.data import DatasetCatalog, MetadataCatalog
from transforms3d.euler import mat2euler

cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../../../../"))
sys.path.insert(0, PROJ_ROOT)

from core.deepim.datasets.dataset_factory import register_datasets
from lib.utils.utils import dprint
from lib.pysixd.pose_error import re, te
from lib.pysixd.RT_transform import calc_RT_delta, se3_mul


id2obj = {
    1: "ape",
    2: "benchvise",
    # 3: "bowl",
    4: "camera",
    5: "can",
    6: "cat",
    # 7: "cup",
    8: "driller",
    9: "duck",
    10: "eggbox",
    11: "glue",
    12: "holepuncher",
    13: "iron",
    14: "lamp",
    15: "phone",
}
obj_num = len(id2obj)
obj2id = {_name: _id for _id, _name in id2obj.items()}


def main():
    init_pose_path = osp.join(
        PROJ_ROOT,
        "datasets/BOP_DATASETS/lm/test/init_poses/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e_so.json",
    )
    dset_name = "lm_13_test"
    print(dset_name)
    register_datasets([dset_name])

    meta = MetadataCatalog.get(dset_name)
    print("MetadataCatalog: ", meta)
    objs = meta.objs

    dset_dicts = DatasetCatalog.get(dset_name)
    scene_im_id_to_gt_index = {d["scene_im_id"]: i for i, d in enumerate(dset_dicts)}

    init_results = mmcv.load(init_pose_path)
    r_errors = {obj_name: [] for obj_name in obj2id}
    euler_x_errors = {obj_name: [] for obj_name in obj2id}
    euler_y_errors = {obj_name: [] for obj_name in obj2id}
    euler_z_errors = {obj_name: [] for obj_name in obj2id}
    t_errors = {obj_name: [] for obj_name in obj2id}
    tx_errors = {obj_name: [] for obj_name in obj2id}
    ty_errors = {obj_name: [] for obj_name in obj2id}
    tz_errors = {obj_name: [] for obj_name in obj2id}

    for scene_im_id, init_res in tqdm(init_results.items()):
        if scene_im_id not in scene_im_id_to_gt_index:
            dprint("{} not in gt dicts".format(scene_im_id))
        gt_idx = scene_im_id_to_gt_index[scene_im_id]
        gt_dict = dset_dicts[gt_idx]
        gt_annos = gt_dict["annotations"]
        for pred in init_res:
            pred_obj_id = pred["obj_id"]
            pred_obj_name = id2obj[pred_obj_id]
            for gt_anno in gt_annos:
                gt_label = gt_anno["category_id"]
                gt_obj = objs[gt_label]
                gt_obj_id = obj2id[gt_obj]
                if pred_obj_id == gt_obj_id:
                    gt_pose = gt_anno["pose"]
                    break
            pred_pose = np.array(pred["pose_est"])
            if pred_obj_name in ["eggbox", "glue"]:
                r_error = re(pred_pose[:3, :3], gt_pose[:3, :3])
                if r_error > 90:
                    RT_z = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0]])
                    pred_pose = se3_mul(pred_pose, RT_z)

            R_delta = np.dot(pred_pose[:3, :3].transpose(), gt_pose[:3, :3])
            euler_delta = mat2euler(R_delta)

            # compute errors
            r_error = re(pred_pose[:3, :3], gt_pose[:3, :3])
            euler_x_error = np.abs(euler_delta[0] * 180 / math.pi)
            euler_y_error = np.abs(euler_delta[1] * 180 / math.pi)
            euler_z_error = np.abs(euler_delta[2] * 180 / math.pi)

            t_error = te(pred_pose[:3, 3], gt_pose[:3, 3])
            tx_error = pred_pose[:3, 0] - gt_pose[:3, 0]
            ty_error = pred_pose[:3, 1] - gt_pose[:3, 1]
            tz_error = pred_pose[:3, 2] - gt_pose[:3, 2]

            # record errors
            r_errors[pred_obj_name].append(r_error)
            euler_x_errors[pred_obj_name].append(euler_x_error)
            euler_y_errors[pred_obj_name].append(euler_y_error)
            euler_z_errors[pred_obj_name].append(euler_z_error)
            t_errors[pred_obj_name].append(t_error)
            tx_errors[pred_obj_name].append(tx_error)
            ty_errors[pred_obj_name].append(ty_error)
            tz_errors[pred_obj_name].append(tz_error)

    # summarize
    for obj_name in obj2id:
        print(obj_name)
        cur_r_errors = np.array(r_errors[obj_name])
        cur_euler_x_errors = np.array(euler_x_errors[obj_name])
        cur_euler_y_errors = np.array(euler_y_errors[obj_name])
        cur_euler_z_errors = np.array(euler_z_errors[obj_name])
        cur_t_errors = np.array(t_errors[obj_name])
        cur_tx_errors = np.array(tx_errors[obj_name])
        cur_ty_errors = np.array(ty_errors[obj_name])
        cur_tz_errors = np.array(tz_errors[obj_name])
        print(
            "r error, mean: {} std: {} min: {} max: {} median: {}".format(
                cur_r_errors.mean(),
                cur_r_errors.std(),
                cur_r_errors.min(),
                cur_r_errors.max(),
                np.median(cur_r_errors),
            )
        )
        print(
            "euler_x error, mean: {} std: {} min: {} max: {} median: {}".format(
                cur_euler_x_errors.mean(),
                cur_euler_x_errors.std(),
                cur_euler_x_errors.min(),
                cur_euler_x_errors.max(),
                np.median(cur_euler_x_errors),
            )
        )
        print(
            "euler_y error, mean: {} std: {} min: {} max: {} median: {}".format(
                cur_euler_y_errors.mean(),
                cur_euler_y_errors.std(),
                cur_euler_y_errors.min(),
                cur_euler_y_errors.max(),
                np.median(cur_euler_y_errors),
            )
        )
        print(
            "euler_z error, mean: {} std: {} min: {} max: {} median: {}".format(
                cur_euler_z_errors.mean(),
                cur_euler_z_errors.std(),
                cur_euler_z_errors.min(),
                cur_euler_z_errors.max(),
                np.median(cur_euler_z_errors),
            )
        )
        print(
            "t error, mean: {} std: {} min: {} max: {} median: {}".format(
                cur_t_errors.mean(),
                cur_t_errors.std(),
                cur_t_errors.min(),
                cur_t_errors.max(),
                np.median(cur_t_errors),
            )
        )
        print(
            "tx error, mean: {} std: {} min: {} max: {} median: {}".format(
                cur_tx_errors.mean(),
                cur_tx_errors.std(),
                cur_tx_errors.min(),
                cur_tx_errors.max(),
                np.median(cur_tx_errors),
            )
        )
        print(
            "ty error, mean: {} std: {} min: {} max: {} median: {}".format(
                cur_ty_errors.mean(),
                cur_ty_errors.std(),
                cur_ty_errors.min(),
                cur_ty_errors.max(),
                np.median(cur_ty_errors),
            )
        )
        print(
            "tz error, mean: {} std: {} min: {} max: {} median: {}".format(
                cur_tz_errors.mean(),
                cur_tz_errors.std(),
                cur_tz_errors.min(),
                cur_tz_errors.max(),
                np.median(cur_tz_errors),
            )
        )
        # Visualize distributions.
        plt.figure(dpi=200)
        row = 2
        col = 4
        bins = 100
        font_size = 8
        matplotlib.rcParams["xtick.labelsize"] = 5
        matplotlib.rcParams["ytick.labelsize"] = 5

        plt.subplot(row, col, 1)
        plt.hist(cur_r_errors, bins=bins)
        plt.title("r error", fontsize=font_size)

        plt.subplot(row, col, 2)
        plt.hist(cur_euler_x_errors, bins=bins)
        plt.title("euler x error", fontsize=font_size)

        plt.subplot(row, col, 3)
        plt.hist(cur_euler_y_errors, bins=bins)
        plt.title("euler y error", fontsize=font_size)

        plt.subplot(row, col, 4)
        plt.hist(cur_euler_z_errors, bins=bins)
        plt.title("euler z error", fontsize=font_size)

        plt.subplot(row, col, 5)
        plt.hist(cur_t_errors, bins=bins)
        plt.title("t error", fontsize=font_size)

        plt.subplot(row, col, 6)
        plt.hist(cur_tx_errors, bins=bins)
        plt.title("tx error", fontsize=font_size)

        plt.subplot(row, col, 7)
        plt.hist(cur_ty_errors, bins=bins)
        plt.title("ty error", fontsize=font_size)

        plt.subplot(row, col, 8)
        plt.hist(cur_tz_errors, bins=bins)
        plt.title("tz error", fontsize=font_size)

        plt.suptitle("{}".format(obj_name), fontsize=font_size)
        plt.show()


if __name__ == "__main__":
    main()
