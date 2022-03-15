import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
import os.path as osp
import sys
import numpy as np
import mmcv
import cv2
import torch
from detectron2.data import DatasetCatalog, MetadataCatalog
from tqdm import tqdm

cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../../../../"))
sys.path.insert(0, PROJ_ROOT)
from lib.egl_renderer.egl_renderer_v3 import EGLRenderer
from lib.pysixd import inout, misc
from lib.vis_utils.image import grid_show
from lib.utils.mask_utils import cocosegm2mask, batch_dice_score
from lib.utils.utils import dprint, iprint, wprint
from lib.pysixd.pose_error import calc_rt_dist_m
from core.gdrn_modeling.datasets.dataset_factory import register_datasets
import ref


id2obj = {
    1: "ape",
    #  2: 'benchvise',
    #  3: 'bowl',
    #  4: 'camera',
    5: "can",
    6: "cat",
    #  7: 'cup',
    8: "driller",
    9: "duck",
    10: "eggbox",
    11: "glue",
    12: "holepuncher",
    #  13: 'iron',
    #  14: 'lamp',
    #  15: 'phone'
}
obj_num = len(id2obj)
obj2id = {_name: _id for _id, _name in id2obj.items()}

IM_H = 480
IM_W = 640
near = 0.01
far = 6.5
K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])

model_dir = osp.normpath(osp.join(PROJ_ROOT, "datasets/BOP_DATASETS/lmo/models"))
obj_ids = [_id for _id in id2obj]
model_paths = [osp.join(model_dir, f"obj_{cls_idx:06d}.ply") for cls_idx in id2obj]
texture_paths = None


if __name__ == "__main__":
    lb_res_root = "output/gdrn/lmoPbrSO/resnest50d_online_AugCosyAAEGray_mlBCE_DoubleMask_lmo_pbr_100e/"
    # cd output/gdrn/lmoPbrSO/resnest50d_online_AugCosyAAEGray_mlBCE_DoubleMask_lmo_pbr_100e/; ls **/**/lmo_NoBopTest_train/results.pkl -1; cd -
    # cd output/gdrn/lmoPbrSO/resnest50d_online_AugCosyAAEGray_mlBCE_DoubleMask_lmo_pbr_100e/; ls **/**/lmo_NoBopTest_train/results_color_aug.pkl -1; cd -
    ss_res_root = "output/self6dpp/ssLMO/ss_mlBCE_MaskFull_PredFull_lr1e_5_lower_woCenter_refinePM10_lmoNoBopTest/"
    ub_res_root = "output/gdrn/lmoRealPbrSO/resnest50d_online_AugCosyAAEGray_mlBCE_DoubleMask_lmoRealNBPbr_100e_SO/"
    # cd output/gdrn/lmoRealPbrSO/resnest50d_online_AugCosyAAEGray_mlBCE_DoubleMask_lmoRealNBPbr_100e_SO/; ls **/**/lmo_NoBopTest_train/results.pkl -1; cd -
    lb_path_names = [
        "ape/inference_model_final_wo_optim-f0ef90df/lmo_NoBopTest_train/results.pkl",
        "can/inference_model_final_wo_optim-ea5b9c78/lmo_NoBopTest_train/results.pkl",
        "cat/inference_model_final_wo_optim-9931aeed/lmo_NoBopTest_train/results.pkl",
        "driller/inference_model_final_wo_optim-bded40f0/lmo_NoBopTest_train/results.pkl",
        "duck/inference_model_final_wo_optim-3cc3dbe6/lmo_NoBopTest_train/results.pkl",
        "eggbox/inference_model_final_wo_optim-817002cd/lmo_NoBopTest_train/results.pkl",
        "glue/inference_model_final_wo_optim-0b8a2e73/lmo_NoBopTest_train/results.pkl",
        "holepuncher/inference_model_final_wo_optim-c98281c9/lmo_NoBopTest_train/results.pkl",
    ]
    lb_aug_path_names = [
        "ape/inference_model_final_wo_optim-f0ef90df/lmo_NoBopTest_train/results_color_aug.pkl",
        "can/inference_model_final_wo_optim-ea5b9c78/lmo_NoBopTest_train/results_color_aug.pkl",
        "cat/inference_model_final_wo_optim-9931aeed/lmo_NoBopTest_train/results_color_aug.pkl",
        "driller/inference_model_final_wo_optim-bded40f0/lmo_NoBopTest_train/results_color_aug.pkl",
        "duck/inference_model_final_wo_optim-3cc3dbe6/lmo_NoBopTest_train/results_color_aug.pkl",
        "eggbox/inference_model_final_wo_optim-817002cd/lmo_NoBopTest_train/results_color_aug.pkl",
        "glue/inference_model_final_wo_optim-0b8a2e73/lmo_NoBopTest_train/results_color_aug.pkl",
        "holepuncher/inference_model_final_wo_optim-c98281c9/lmo_NoBopTest_train/results_color_aug.pkl",
    ]
    # --------------
    ss_path_names = [
        "ape/inference_model_final_wo_optim-57c901fc/lmo_NoBopTest_ape_train/results.pkl",
        "can/inference_model_final_wo_optim-db96d3dc/lmo_NoBopTest_can_train/results.pkl",
        "cat/inference_model_final_wo_optim-d27458fb/lmo_NoBopTest_cat_train/results.pkl",
        "driller/inference_model_final_wo_optim-64eec6b2/lmo_NoBopTest_driller_train/results.pkl",
        "duck/inference_model_final_wo_optim-5c6dc578/lmo_NoBopTest_duck_train/results.pkl",
        "eggbox/inference_model_final_wo_optim-45db2b71/lmo_NoBopTest_eggbox_train/results.pkl",
        "glue/inference_model_final_wo_optim-60598376/lmo_NoBopTest_glue_train/results.pkl",
        "holepuncher/inference_model_final_wo_optim-a8606013/lmo_NoBopTest_holepuncher_train/results.pkl",
    ]
    ub_path_names = [
        "ape/inference_model_final_wo_optim-5d6676ae/lmo_NoBopTest_train/results.pkl",
        "can/inference_model_final_wo_optim-e2869b7c/lmo_NoBopTest_train/results.pkl",
        "cat/inference_model_final_wo_optim-4c0c4c63/lmo_NoBopTest_train/results.pkl",
        "driller/inference_model_final_wo_optim-fa6bdff7/lmo_NoBopTest_train/results.pkl",
        "duck/inference_model_final_wo_optim-6d1dcf53/lmo_NoBopTest_train/results.pkl",
        "eggbox/inference_model_final_wo_optim-be38dae4/lmo_NoBopTest_train/results.pkl",
        "glue/inference_model_final_wo_optim-40d82ce9/lmo_NoBopTest_train/results.pkl",
        "holepuncher/inference_model_final_wo_optim-a7b6e476/lmo_NoBopTest_train/results.pkl",
    ]

    dset_name = "lmo_NoBopTest_train"
    iprint(dset_name)
    register_datasets([dset_name])

    meta = MetadataCatalog.get(dset_name)
    iprint("MetadataCatalog: ", meta)
    objs = meta.objs

    # dset_dicts = DatasetCatalog.get(dset_name)
    # scene_im_id_to_gt_index = {d["scene_im_id"]: i for i, d in enumerate(dset_dicts)}

    # ren = EGLRenderer(
    #     model_paths,
    #     texture_paths=texture_paths,
    #     vertex_scale=0.001,
    #     height=IM_H,
    #     width=IM_W,
    #     znear=near,
    #     zfar=far,
    #     use_cache=True,
    # )
    height = IM_H
    width = IM_W
    # device = "cuda"
    # image_tensor = torch.cuda.FloatTensor(height, width, 4, device=device).detach()
    # seg_tensor = torch.cuda.FloatTensor(height, width, 4, device=device).detach()
    # pc_cam_tensor = torch.cuda.FloatTensor(height, width, 4, device=device).detach()

    merged_results = {}  # key is scene_im_id, each contains a list of instance dicts
    merged_save_path = osp.join(ss_res_root, "merged-NoBopTest/merged_lb_lbAug_ss_ub_results.pkl")
    # output/self6dpp/ssLMO/ss_mlBCE_MaskFull_PredFull_PBR05_lr1e-5_woCenter_lmoNoBopTest/merged-NoBopTest/merged_lb_lbAug_ss_ub_results.pkl
    if osp.exists(merged_save_path):
        wprint("merged_results_path exists, overriding!!!")

    for i, obj_name in enumerate(obj2id):
        iprint(obj_name)

        lb_res_path = osp.join(lb_res_root, lb_path_names[i])
        assert osp.exists(lb_res_path), lb_res_path
        assert obj_name in lb_path_names[i]

        lb_aug_res_path = osp.join(lb_res_root, lb_aug_path_names[i])
        assert osp.exists(lb_aug_res_path), lb_aug_res_path
        assert obj_name in lb_aug_path_names[i]

        ss_res_path = osp.join(ss_res_root, ss_path_names[i])
        assert osp.exists(ss_res_path), ss_res_path
        assert obj_name in ss_path_names[i]

        ub_res_path = osp.join(ub_res_root, ub_path_names[i])
        assert osp.exists(ub_res_path), ub_res_path
        assert obj_name in ub_path_names[i]

        # pkl  scene_im_id key, list of preds
        lb_results = mmcv.load(lb_res_path)
        lb_aug_results = mmcv.load(lb_aug_res_path)

        ss_results = mmcv.load(ss_res_path)
        ub_results = mmcv.load(ub_res_path)

        for scene_im_id in tqdm(lb_results):
            if scene_im_id not in ss_results:
                # __import__("ipdb").set_trace()
                wprint("{} not in {} ss_results".format(scene_im_id, obj_name))
                continue
            lb_res_list = lb_results[scene_im_id]
            lb_aug_res_list = lb_aug_results[scene_im_id]
            ss_res_list = ss_results[scene_im_id]
            ub_res_list = ub_results[scene_im_id]

            if scene_im_id not in merged_results:
                merged_results[scene_im_id] = []

            for lb_res, lb_aug_res, ss_res, ub_res in zip(lb_res_list, lb_aug_res_list, ss_res_list, ub_res_list):
                # common results ------------------
                obj_id = lb_res["obj_id"]
                score = lb_res["score"]
                bbox_est = lb_res["bbox_est"]
                cur_merged = {
                    "obj_id": obj_id,
                    "score": score,
                    "bbox_est": bbox_est,
                }

                # lb results -----------------
                lb_pose = np.zeros((3, 4), dtype=np.float32)
                lb_pose[:3, :3] = lb_res["R"]
                lb_pose[:3, 3] = lb_res["t"]

                cur_merged["lb_pose"] = lb_pose
                cur_merged["lb_mask_vis"] = lb_res["mask"]

                # lb_mask_vis = cocosegm2mask(lb_res["mask"], IM_H, IM_W)
                if "full_mask" in lb_res:
                    # lb_mask_full = cocosegm2mask(lb_res["full_mask"], IM_H, IM_W)
                    cur_merged["lb_mask_full"] = lb_res["full_mask"]

                # lb aug results --------------------
                lb_aug_pose = np.zeros((3, 4), dtype=np.float32)
                lb_aug_pose[:3, :3] = lb_aug_res["R"]
                lb_aug_pose[:3, 3] = lb_aug_res["t"]

                cur_merged["lb_aug_pose"] = lb_aug_pose
                cur_merged["lb_aug_mask_vis"] = lb_aug_res["mask"]

                # lb_aug_mask_vis = cocosegm2mask(lb_aug_res['mask'], IM_H, IM_W)
                if "full_mask" in lb_aug_res:
                    # lb_aug_mask_full = cocosegm2mask(lb_aug_res['full_mask'], IM_H, IM_W)
                    cur_merged["lb_aug_mask_full"] = lb_aug_res["full_mask"]

                # ss results ------------------------
                ss_pose = np.zeros((3, 4), dtype=np.float32)
                ss_pose[:3, :3] = ss_res["R"]
                ss_pose[:3, 3] = ss_res["t"]

                cur_merged["ss_pose"] = ss_pose
                cur_merged["ss_mask_vis"] = ss_res["mask"]

                # ss_mask_vis = cocosegm2mask(ss_res["mask"], IM_H, IM_W)
                if "full_mask" in ss_res:
                    # ss_mask_full = cocosegm2mask(ss_res["full_mask"], IM_H, IM_W)
                    cur_merged["ss_mask_full"] = ss_res["full_mask"]

                # ub results ----------------------------------
                ub_pose = np.zeros((3, 4), dtype=np.float32)
                ub_pose[:3, :3] = ub_res["R"]
                ub_pose[:3, 3] = ub_res["t"]

                cur_merged["ub_pose"] = ub_pose
                cur_merged["ub_mask_vis"] = ub_res["mask"]

                # ub_mask_vis = cocosegm2mask(ub_res["mask"], IM_H, IM_W)
                if "full_mask" in ub_res:
                    # ub_mask_full = cocosegm2mask(ub_res["full_mask"], IM_H, IM_W)
                    cur_merged["ub_mask_full"] = ub_res["full_mask"]

                # -----------------------------------------------------
                merged_results[scene_im_id].append(cur_merged)

    # save merged results
    mmcv.mkdir_or_exist(osp.dirname(merged_save_path))
    mmcv.dump(merged_results, merged_save_path)
    iprint("merged results saved to {}".format(merged_save_path))
