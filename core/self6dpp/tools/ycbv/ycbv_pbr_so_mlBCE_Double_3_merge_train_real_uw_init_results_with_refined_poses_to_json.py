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
    1: "002_master_chef_can",  # [1.3360, -0.5000, 3.5105]
    2: "003_cracker_box",  # [0.5575, 1.7005, 4.8050]
    3: "004_sugar_box",  # [-0.9520, 1.4670, 4.3645]
    4: "005_tomato_soup_can",  # [-0.0240, -1.5270, 8.4035]
    5: "006_mustard_bottle",  # [1.2995, 2.4870, -11.8290]
    6: "007_tuna_fish_can",  # [-0.1565, 0.1150, 4.2625]
    7: "008_pudding_box",  # [1.1645, -4.2015, 3.1190]
    8: "009_gelatin_box",  # [1.4460, -0.5915, 3.6085]
    9: "010_potted_meat_can",  # [2.4195, 0.3075, 8.0715]
    10: "011_banana",  # [-18.6730, 12.1915, -1.4635]
    11: "019_pitcher_base",  # [5.3370, 5.8855, 25.6115]
    12: "021_bleach_cleanser",  # [4.9290, -2.4800, -13.2920]
    13: "024_bowl",  # [-0.2270, 0.7950, -2.9675]
    14: "025_mug",  # [-8.4675, -0.6995, -1.6145]
    15: "035_power_drill",  # [9.0710, 20.9360, -2.1190]
    16: "036_wood_block",  # [1.4265, -2.5305, 17.1890]
    17: "037_scissors",  # [7.0535, -28.1320, 0.0420]
    18: "040_large_marker",  # [0.0460, -2.1040, 0.3500]
    19: "051_large_clamp",  # [10.5180, -1.9640, -0.4745]
    20: "052_extra_large_clamp",  # [-0.3950, -10.4130, 0.1620]
    21: "061_foam_brick",  # [-0.0805, 0.0805, -8.2435]
}
obj_num = len(id2obj)
obj2id = {_name: _id for _id, _name in id2obj.items()}


if __name__ == "__main__":
    new_res_path = osp.join(
        PROJ_ROOT,
        "datasets/BOP_DATASETS/ycbv/test/init_poses/",
        "resnest50d_online_AugCosyAAEGray_mlBCE_DoubleMask_ycbv_pbr_100e_so_GdrnPbrPose_withYolov4PbrBbox_wDeepimPbrPose_ycbv_train_real_uw.json",
    )
    if osp.exists(new_res_path):
        wprint("{} already exists! overriding!".format(new_res_path))

    res_root = "output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/"
    iter_num_test = 4

    pkl_paths = [
        "01_02MasterChefCan/inference_model_final_wo_optim-2de2b4e3/ycbv_002_master_chef_can_train_real_uw/results.pkl",
        "02_03CrackerBox/inference_model_final_wo_optim-41082f8a/ycbv_003_cracker_box_train_real_uw/results.pkl",
        "03_04SugarBox/inference_model_final_wo_optim-e09dec3e/ycbv_004_sugar_box_train_real_uw/results.pkl",
        "04_05TomatoSoupCan/inference_model_final_wo_optim-5641f5d3/ycbv_005_tomato_soup_can_train_real_uw/results.pkl",
        "05_06MustardBottle/inference_model_final_wo_optim-6ce23e94/ycbv_006_mustard_bottle_train_real_uw/results.pkl",
        "06_07TunaFishCan/inference_model_final_wo_optim-0a768962/ycbv_007_tuna_fish_can_train_real_uw/results.pkl",
        "07_08PuddingBox/inference_model_final_wo_optim-f2f2cf73/ycbv_008_pudding_box_train_real_uw/results.pkl",
        "08_09GelatinBox/inference_model_final_wo_optim-a303aa1e/ycbv_009_gelatin_box_train_real_uw/results.pkl",
        "09_10PottedMeatCan/inference_model_final_wo_optim-84a56ffd/ycbv_010_potted_meat_can_train_real_uw/results.pkl",
        "10_11Banana/inference_model_final_wo_optim-83947126/ycbv_011_banana_train_real_uw/results.pkl",
        "11_19PitcherBase/inference_model_final_wo_optim-af1c7e62/ycbv_019_pitcher_base_train_real_uw/results.pkl",
        "12_21BleachCleanser/inference_model_final_wo_optim-5d740a46/ycbv_021_bleach_cleanser_train_real_uw/results.pkl",
        "13_24Bowl/inference_model_final_wo_optim-f11815d3/ycbv_024_bowl_train_real_uw/results.pkl",
        "14_25Mug/inference_model_final_wo_optim-e4824065/ycbv_025_mug_train_real_uw/results.pkl",
        "15_35PowerDrill/inference_model_final_wo_optim-30d7d1da/ycbv_035_power_drill_train_real_uw/results.pkl",
        "16_36WoodBlock/inference_model_final_wo_optim-fbb38751/ycbv_036_wood_block_train_real_uw/results.pkl",
        "17_37Scissors/inference_model_final_wo_optim-5068c6bb/ycbv_037_scissors_train_real_uw/results.pkl",
        "18_40LargeMarker/inference_model_final_wo_optim-e8d5867c/ycbv_040_large_marker_train_real_uw/results.pkl",
        "19_51LargeClamp/inference_model_final_wo_optim-1ea79b34/ycbv_051_large_clamp_train_real_uw/results.pkl",
        "20_52ExtraLargeClamp/inference_model_final_wo_optim-cb595297/ycbv_052_extra_large_clamp_train_real_uw/results.pkl",
        "21_61FoamBrick/inference_model_final_wo_optim-d3757ca1/ycbv_061_foam_brick_train_real_uw/results.pkl",
    ]
    obj_names = [obj for obj in obj2id]

    new_res_dict = {}
    for obj_name, pred_name in zip(obj_names, pkl_paths):
        assert obj_name in pred_name, "{} not in {}".format(obj_name, pred_name)

        pred_path = osp.join(res_root, pred_name)
        assert osp.exists(pred_path), pred_path
        iprint(obj_name, pred_path)

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

    inout.save_json(new_res_path, new_res_dict)
    iprint()
    iprint("new result path: {}".format(new_res_path))
