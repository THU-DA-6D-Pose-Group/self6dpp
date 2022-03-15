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
    out_root = "output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/"
    pred_paths = [
        "ape/inference_model_final_wo_optim-e8c99c96/lm_real_ape_all/results.pkl",
        "benchvise/inference_model_final_wo_optim-85b3563e/lm_real_benchvise_all/results.pkl",
        "camera/inference_model_final_wo_optim-1b281dbe/lm_real_camera_all/results.pkl",
        "can/inference_model_final_wo_optim-53ea56ee/lm_real_can_all/results.pkl",
        "cat/inference_model_final_wo_optim-f38cfafd/lm_real_cat_all/results.pkl",
        "driller/inference_model_final_wo_optim-4cfc7d64/lm_real_driller_all/results.pkl",
        "duck/inference_model_final_wo_optim-0bde58bb/lm_real_duck_all/results.pkl",
        "eggbox_Rsym/inference_model_final_wo_optim-d0656ca7/lm_real_eggbox_all/results.pkl",
        "glue_Rsym/inference_model_final_wo_optim-324d8f16/lm_real_glue_all/results.pkl",
        "holepuncher/inference_model_final_wo_optim-eab19662/lm_real_holepuncher_all/results.pkl",
        "iron/inference_model_final_wo_optim-025a740e/lm_real_iron_all/results.pkl",
        "lamp/inference_model_final_wo_optim-34042758/lm_real_lamp_all/results.pkl",
        "phone/inference_model_final_wo_optim-525a29f8/lm_real_phone_all/results.pkl",
    ]
    obj_names = [obj for obj in obj2id]

    new_res_dict = {}
    for obj_name, pred_name in zip(obj_names, pred_paths):
        assert obj_name in pred_name, "{} not in {}".format(obj_name, pred_name)

        pred_path = osp.join(out_root, pred_name)
        assert osp.exists(pred_path), pred_path
        print(obj_name, pred_path)

        # pkl  scene_im_id key, list of preds
        preds = mmcv.load(pred_path)

        for scene_im_id, pred_list in preds.items():
            for pred in pred_list:
                obj_id = pred["obj_id"]
                score = pred["score"]
                bbox_est = pred["bbox_est"]  # xyxy
                bbox_est_xywh = xyxy_to_xywh(bbox_est)

                R_est = pred["R"]
                t_est = pred["t"]
                pose_est = np.hstack([R_est, t_est.reshape(3, 1)])
                cur_new_res = {
                    "obj_id": obj_id,
                    "score": float(score),
                    "bbox_est": bbox_est_xywh.tolist(),
                    "pose_est": pose_est.tolist(),
                }
                if scene_im_id not in new_res_dict:
                    new_res_dict[scene_im_id] = []
                new_res_dict[scene_im_id].append(cur_new_res)

    new_res_path = osp.join(
        PROJ_ROOT,
        "datasets/BOP_DATASETS/lm/test/init_poses/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e_so_with_faster_rcnn101_pbr_bbox.json",
    )
    inout.save_json(new_res_path, new_res_dict)
    print()
    print("new result path: {}".format(new_res_path))
