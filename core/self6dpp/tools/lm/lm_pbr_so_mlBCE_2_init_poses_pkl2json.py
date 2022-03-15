import os.path as osp
import sys
import numpy as np
import mmcv

cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../../../../"))
sys.path.insert(0, PROJ_ROOT)
from lib.pysixd import inout, misc


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
    pred_paths_dict = {
        "ape": osp.join(
            out_root,
            "ape/inference_model_final_wo_optim-e8c99c96/lm_real_ape_test/resnest50d-a6-AugCosyAAEGary-BG05-mlBCE-lm-pbr-100e-ape-test_lm_real_ape_test_preds.pkl",
        ),
        "benchvise": osp.join(
            out_root,
            "benchvise/inference_model_final_wo_optim-85b3563e/lm_real_benchvise_test/resnest50d-a6-AugCosyAAEGary-BG05-mlBCE-lm-pbr-100e-benchvise-test_lm_real_benchvise_test_preds.pkl",
        ),
        "camera": osp.join(
            out_root,
            "camera/inference_model_final_wo_optim-1b281dbe/lm_real_camera_test/resnest50d-a6-AugCosyAAEGary-BG05-mlBCE-lm-pbr-100e-camera-test_lm_real_camera_test_preds.pkl",
        ),
        "can": osp.join(
            out_root,
            "can/inference_model_final_wo_optim-53ea56ee/lm_real_can_test/resnest50d-a6-AugCosyAAEGary-BG05-mlBCE-lm-pbr-100e-can-test_lm_real_can_test_preds.pkl",
        ),
        "cat": osp.join(
            out_root,
            "cat/inference_model_final_wo_optim-f38cfafd/lm_real_cat_test/resnest50d-a6-AugCosyAAEGary-BG05-mlBCE-lm-pbr-100e-cat-test_lm_real_cat_test_preds.pkl",
        ),
        "driller": osp.join(
            out_root,
            "driller/inference_model_final_wo_optim-4cfc7d64/lm_real_driller_test/resnest50d-a6-AugCosyAAEGary-BG05-mlBCE-lm-pbr-100e-driller-test_lm_real_driller_test_preds.pkl",
        ),
        "duck": osp.join(
            out_root,
            "duck/inference_model_final_wo_optim-0bde58bb/lm_real_duck_test/resnest50d-a6-AugCosyAAEGary-BG05-mlBCE-lm-pbr-100e-duck-test_lm_real_duck_test_preds.pkl",
        ),
        "eggbox": osp.join(
            out_root,
            "eggbox_Rsym/inference_model_final_wo_optim-d0656ca7/lm_real_eggbox_test/resnest50d-a6-AugCosyAAEGary-BG05-mlBCE-lm-pbr-100e-eggbox-Rsym-test_lm_real_eggbox_test_preds.pkl",
        ),
        "glue": osp.join(
            out_root,
            "glue_Rsym/inference_model_final_wo_optim-324d8f16/lm_real_glue_test/resnest50d-a6-AugCosyAAEGary-BG05-mlBCE-lm-pbr-100e-glue-Rsym-test_lm_real_glue_test_preds.pkl",
        ),
        "holepuncher": osp.join(
            out_root,
            "holepuncher/inference_model_final_wo_optim-eab19662/lm_real_holepuncher_test/resnest50d-a6-AugCosyAAEGary-BG05-mlBCE-lm-pbr-100e-holepuncher-test_lm_real_holepuncher_test_preds.pkl",
        ),
        "iron": osp.join(
            out_root,
            "iron/inference_model_final_wo_optim-025a740e/lm_real_iron_test/resnest50d-a6-AugCosyAAEGary-BG05-mlBCE-lm-pbr-100e-iron-test_lm_real_iron_test_preds.pkl",
        ),
        "lamp": osp.join(
            out_root,
            "lamp/inference_model_final_wo_optim-34042758/lm_real_lamp_test/resnest50d-a6-AugCosyAAEGary-BG05-mlBCE-lm-pbr-100e-lamp-test_lm_real_lamp_test_preds.pkl",
        ),
        "phone": osp.join(
            out_root,
            "phone/inference_model_final_wo_optim-525a29f8/lm_real_phone_test/resnest50d-a6-AugCosyAAEGary-BG05-mlBCE-lm-pbr-100e-phone-test_lm_real_phone_test_preds.pkl",
        ),
    }
    new_res_dict = {}
    for obj_name, pred_path in pred_paths_dict.items():
        print(obj_name, pred_path)
        assert osp.exists(pred_path), pred_path
        preds = mmcv.load(pred_path)
        obj_preds = preds[obj_name]
        for file_path, results in obj_preds.items():
            im_id = int(osp.basename(file_path).split(".")[0])
            scene_id = int(file_path.split("/")[-3])
            scene_im_id = f"{scene_id}/{im_id}"
            R_est = results["R"]
            t_est = results["t"]
            pose_est = np.hstack([R_est, t_est.reshape(3, 1)])
            cur_new_res = {
                "obj_id": obj2id[obj_name],
                "pose_est": pose_est.tolist(),
                "score": 1.0,  # dummy
            }
            if scene_im_id not in new_res_dict:
                new_res_dict[scene_im_id] = []
            new_res_dict[scene_im_id].append(cur_new_res)

    new_res_path = osp.join(
        PROJ_ROOT,
        "datasets/BOP_DATASETS/lm/test/init_poses/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e_so.json",
    )
    inout.save_json(new_res_path, new_res_dict)
    print("new result path: {}".format(new_res_path))
