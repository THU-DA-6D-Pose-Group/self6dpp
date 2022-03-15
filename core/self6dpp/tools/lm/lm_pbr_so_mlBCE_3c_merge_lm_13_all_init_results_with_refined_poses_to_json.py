import os.path as osp
import sys
import numpy as np
import mmcv
from tqdm import tqdm

cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../../../../"))
sys.path.insert(0, PROJ_ROOT)
from lib.pysixd import inout, misc
from lib.utils.bbox_utils import xyxy_to_xywh


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


if __name__ == "__main__":
    res_root = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/"
    iter_num_test = 4

    pkl_paths = [
        "ape/inference_model_final_wo_optim-c5ab39d4/lm_real_ape_all/results.pkl",
        "benchvise/inference_model_final_wo_optim-956eb2eb/lm_real_benchvise_all/results.pkl",
        "camera/inference_model_final_wo_optim-520e2c6d/lm_real_camera_all/results.pkl",
        "can/inference_model_final_wo_optim-29275594/lm_real_can_all/results.pkl",
        "cat/inference_model_final_wo_optim-bd14a802/lm_real_cat_all/results.pkl",
        "driller/inference_model_final_wo_optim-45c5dd36/lm_real_driller_all/results.pkl",
        "duck/inference_model_final_wo_optim-e28d46ee/lm_real_duck_all/results.pkl",
        "eggbox/inference_model_final_wo_optim-bb583866/lm_real_eggbox_all/results.pkl",
        "glue/inference_model_final_wo_optim-8013fa70/lm_real_glue_all/results.pkl",
        "holepuncher/inference_model_final_wo_optim-30e67e31/lm_real_holepuncher_all/results.pkl",
        "iron/inference_model_final_wo_optim-485fd5c3/lm_real_iron_all/results.pkl",
        "lamp/inference_model_final_wo_optim-f169005c/lm_real_lamp_all/results.pkl",
        "phone/inference_model_final_wo_optim-61aad172/lm_real_phone_all/results.pkl",
    ]
    obj_names = [obj for obj in obj2id]

    new_res_dict = {}
    for obj_name, pred_name in zip(obj_names, pkl_paths):
        assert obj_name in pred_name, "{} not in {}".format(obj_name, pred_name)

        pred_path = osp.join(res_root, pred_name)
        assert osp.exists(pred_path), pred_path
        print(obj_name, pred_path)

        # pkl  scene_im_id key, list of preds
        preds = mmcv.load(pred_path)

        for scene_im_id, pred_list in preds.items():
            for pred in pred_list:
                obj_id = pred["obj_id"]
                score = pred["score"]
                bbox_est = pred["bbox_det_xyxy"]  # xyxy
                bbox_est_xywh = xyxy_to_xywh(bbox_est)

                refined_pose = pred["pose_{}".format(iter_num_test)]
                pose_est = pred["pose_0"]
                cur_new_res = {
                    "obj_id": obj_id,
                    "score": float(score),
                    "bbox_est": bbox_est_xywh.tolist(),
                    "pose_est": pose_est.tolist(),
                    "pose_refine": refined_pose.tolist(),
                }
                if scene_im_id not in new_res_dict:
                    new_res_dict[scene_im_id] = []
                new_res_dict[scene_im_id].append(cur_new_res)

    new_res_path = osp.join(
        PROJ_ROOT,
        "datasets/BOP_DATASETS/lm/test/init_poses/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e_so_withYolov4PbrBbox_wDeepimPbrPose_lm_13_all.json",
    )
    inout.save_json(new_res_path, new_res_dict)
    print()
    print("new result path: {}".format(new_res_path))
