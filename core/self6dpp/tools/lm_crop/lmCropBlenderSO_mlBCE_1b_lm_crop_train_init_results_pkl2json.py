import os.path as osp
import sys
import numpy as np
import mmcv
from tqdm import tqdm
from functools import cmp_to_key

cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../../../../"))
sys.path.insert(0, PROJ_ROOT)
from lib.pysixd import inout, misc
from lib.utils.bbox_utils import xyxy_to_xywh
from lib.utils.utils import iprint, wprint


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
    # 10: "eggbox",
    # 11: "glue",
    12: "holepuncher",
    13: "iron",
    14: "lamp",
    15: "phone",
}
obj_num = len(id2obj)
obj2id = {_name: _id for _id, _name in id2obj.items()}


if __name__ == "__main__":
    new_res_path = osp.join(
        PROJ_ROOT,
        "datasets/BOP_DATASETS/lm/test/init_poses/",
        "resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e_so_lmCropTrain_GdrnPose_with_bbox_crop.json",
    )
    if osp.exists(new_res_path):
        wprint("{} already exists! overriding!".format(new_res_path))

    out_root = "output/gdrn/lm_crop_blender/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e"
    pred_paths = [
        "ape/inference_model_final_wo_optim-ab325295/lm_crop_ape_train/results.pkl",
        "benchvise/inference_model_final_wo_optim-cba7da41/lm_crop_benchvise_train/results.pkl",
        "camera/inference_model_final_wo_optim-5055333e/lm_crop_camera_train/results.pkl",
        "can/inference_model_final_wo_optim-889ae475/lm_crop_can_train/results.pkl",
        "cat/inference_model_final_wo_optim-c754e587/lm_crop_cat_train/results.pkl",
        "driller/inference_model_final_wo_optim-c1cc7169/lm_crop_driller_train/results.pkl",
        "duck/inference_model_final_wo_optim-084c3364/lm_crop_duck_train/results.pkl",
        "holepuncher/inference_model_final_wo_optim-86d52ab7/lm_crop_holepuncher_train/results.pkl",
        "iron/inference_model_final_wo_optim-6bdb61b0/lm_crop_iron_train/results.pkl",
        "lamp/inference_model_final_wo_optim-bf52eba3/lm_crop_lamp_train/results.pkl",
        "phone/inference_model_final_wo_optim-4172540b/lm_crop_phone_train/results.pkl",
    ]
    obj_names = [obj for obj in obj2id]

    new_res_dict = {}
    for obj_name, pred_name in zip(obj_names, pred_paths):
        assert obj_name in pred_name, "{} not in {}".format(obj_name, pred_name)

        pred_path = osp.join(out_root, pred_name)
        assert osp.exists(pred_path), pred_path
        iprint(obj_name, pred_path)

        # pkl  scene_im_id key, list of preds
        preds = mmcv.load(pred_path)

        for scene_im_id, pred_list in preds.items():
            for pred in pred_list:
                obj_id = pred["obj_id"]
                score = pred["score"]
                bbox_crop = pred["bbox_crop"]  # xyxy
                bbox_crop_xywh = xyxy_to_xywh(bbox_crop)

                R_est = pred["R"]
                t_est = pred["t"]
                pose_est = np.hstack([R_est, t_est.reshape(3, 1)])
                cur_new_res = {
                    "obj_id": obj_id,
                    "score": float(score),
                    # "bbox_crop": bbox_crop_xywh.tolist(),
                    "bbox_est": bbox_crop_xywh.tolist(),
                    "pose_est": pose_est.tolist(),
                }
                if scene_im_id not in new_res_dict:
                    new_res_dict[scene_im_id] = []
                new_res_dict[scene_im_id].append(cur_new_res)

    def mycmp(x, y):
        # compare two scene_im_id
        x_scene_id = int(x[0].split("/")[0])
        y_scene_id = int(y[0].split("/")[0])
        if x_scene_id == y_scene_id:
            x_im_id = int(x[0].split("/")[1])
            y_im_id = int(y[0].split("/")[1])
            return x_im_id - y_im_id
        else:
            return x_scene_id - y_scene_id

    new_res_dict_sorted = dict(sorted(new_res_dict.items(), key=cmp_to_key(mycmp)))
    inout.save_json(new_res_path, new_res_dict_sorted)
    iprint()
    iprint("new result path: {}".format(new_res_path))
