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
    pred_paths_dict = {
        "ape": "output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAE_BG05_lm_pbr_100e_ape/inference_model_final_wo_optim-bd176613/lm_real_ape_test/resnest50d-a6-AugCosyAAE-BG05-lm-pbr-100e-ape-test_lm_real_ape_test_preds.pkl",
        "benchvise": "output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAE_BG05_lm_pbr_100e_benchvise/inference_model_final_wo_optim-e9887b86/lm_real_benchvise_test/resnest50d-a6-AugCosyAAE-BG05-lm-pbr-100e-benchvise-test_lm_real_benchvise_test_preds.pkl",
        "camera": "output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAE_BG05_lm_pbr_100e_camera/inference_model_final_wo_optim-3003c8c2/lm_real_camera_test/resnest50d-a6-AugCosyAAE-BG05-lm-pbr-100e-camera-test_lm_real_camera_test_preds.pkl",
        "can": "output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAE_BG05_lm_pbr_100e_can/inference_model_final_wo_optim-4137c7d4/lm_real_can_test/resnest50d-a6-AugCosyAAE-BG05-lm-pbr-100e-can-test_lm_real_can_test_preds.pkl",
        "cat": "output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAE_BG05_lm_pbr_100e_cat/inference_model_final_wo_optim-0e4a79b0/lm_real_cat_test/resnest50d-a6-AugCosyAAE-BG05-lm-pbr-100e-cat-test_lm_real_cat_test_preds.pkl",
        "driller": "output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAE_BG05_lm_pbr_100e_driller/inference_model_final_wo_optim-89726263/lm_real_driller_test/resnest50d-a6-AugCosyAAE-BG05-lm-pbr-100e-driller-test_lm_real_driller_test_preds.pkl",
        "duck": "output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAE_BG05_lm_pbr_100e_duck/inference_model_final_wo_optim-ad82c594/lm_real_duck_test/resnest50d-a6-AugCosyAAE-BG05-lm-pbr-100e-duck-test_lm_real_duck_test_preds.pkl",
        "eggbox": "output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAE_BG05_lm_pbr_100e_eggbox_Rsym/inference_model_final_wo_optim-3fced9ba/lm_real_eggbox_test/resnest50d-a6-AugCosyAAE-BG05-lm-pbr-100e-eggbox-Rsym-test_lm_real_eggbox_test_preds.pkl",
        "glue": "output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAE_BG05_lm_pbr_100e_glue_Rsym/inference_model_final_wo_optim-9e9633a5/lm_real_glue_test/resnest50d-a6-AugCosyAAE-BG05-lm-pbr-100e-glue-Rsym-test_lm_real_glue_test_preds.pkl",
        "holepuncher": "output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAE_BG05_lm_pbr_100e_holepuncher/inference_model_final_wo_optim-55a67aa9/lm_real_holepuncher_test/resnest50d-a6-AugCosyAAE-BG05-lm-pbr-100e-holepuncher-test_lm_real_holepuncher_test_preds.pkl",
        "iron": "output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAE_BG05_lm_pbr_100e_iron/inference_model_final_wo_optim-32f1a16d/lm_real_iron_test/resnest50d-a6-AugCosyAAE-BG05-lm-pbr-100e-iron-test_lm_real_iron_test_preds.pkl",
        "lamp": "output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAE_BG05_lm_pbr_100e_lamp/inference_model_final_wo_optim-cb26bc72/lm_real_lamp_test/resnest50d-a6-AugCosyAAE-BG05-lm-pbr-100e-lamp-test_lm_real_lamp_test_preds.pkl",
        "phone": "output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAE_BG05_lm_pbr_100e_phone/inference_model_final_wo_optim-92a56dd7/lm_real_phone_test/resnest50d-a6-AugCosyAAE-BG05-lm-pbr-100e-phone-test_lm_real_phone_test_preds.pkl",
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
                "score": 1.0,
            }  # dummy
            if scene_im_id not in new_res_dict:
                new_res_dict[scene_im_id] = []
            new_res_dict[scene_im_id].append(cur_new_res)

    new_res_path = osp.join(
        PROJ_ROOT,
        "datasets/BOP_DATASETS/lm/test/init_poses/resnest50d_a6_AugCosyAAE_BG05_lm_pbr_100e_so.json",
    )
    inout.save_json(new_res_path, new_res_dict)
    print("new result path: {}".format(new_res_path))
