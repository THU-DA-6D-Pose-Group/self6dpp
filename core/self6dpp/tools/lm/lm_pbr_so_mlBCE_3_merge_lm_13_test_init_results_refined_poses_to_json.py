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
from lib.utils.utils import dprint
from lib.pysixd.pose_error import calc_rt_dist_m
from core.gdrn_modeling.datasets.dataset_factory import register_datasets
import ref


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

IM_H = 480
IM_W = 640
near = 0.01
far = 6.5
K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])


lm_model_dir = osp.normpath(osp.join(PROJ_ROOT, "datasets/BOP_DATASETS/lm/models"))
obj_ids = [_id for _id in id2obj]
model_paths = [osp.join(lm_model_dir, f"obj_{cls_idx:06d}.ply") for cls_idx in id2obj]
texture_paths = None

VIS = False


if __name__ == "__main__":
    # NOTE: for lm real test
    init_res_root = "output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/"
    refined_res_root = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/"

    init_refine_path_names = {
        "ape": (
            "ape/inference_model_final_wo_optim-e8c99c96/lm_real_ape_test/results.pkl",
            "ape/inference_model_final/lm_real_ape_test/FlowNet512-1.5AugCosyAAEGray-AggressiveV2-Flat-Pbr-01-ape-test_lm_real_ape_test_preds.pkl",
        ),
        "benchvise": (
            "benchvise/inference_model_final_wo_optim-85b3563e/lm_real_benchvise_test/results.pkl",
            "benchvise/inference_/lm_real_benchvise_test/FlowNet512-1.5AugCosyAAEGray-AggressiveV2-Flat-Pbr-02-benchvise_lm_real_benchvise_test_preds.pkl",
        ),
        "camera": (
            "camera/inference_model_final_wo_optim-1b281dbe/lm_real_camera_test/results.pkl",
            "camera/inference_/lm_real_camera_test/FlowNet512-1.5AugCosyAAEGray-AggressiveV2-Flat-Pbr-04-camera_lm_real_camera_test_preds.pkl",
        ),
        "can": (
            "can/inference_model_final_wo_optim-53ea56ee/lm_real_can_test/results.pkl",
            "can/inference_model_final/lm_real_can_test/FlowNet512-1.5AugCosyAAEGray-AggressiveV2-Flat-Pbr-05-can-test_lm_real_can_test_preds.pkl",
        ),
        "cat": (
            "cat/inference_model_final_wo_optim-f38cfafd/lm_real_cat_test/results.pkl",
            "cat/inference_model_final/lm_real_cat_test/FlowNet512-1.5AugCosyAAEGray-AggressiveV2-Flat-Pbr-06-cat-test_lm_real_cat_test_preds.pkl",
        ),
        "driller": (
            "driller/inference_model_final_wo_optim-4cfc7d64/lm_real_driller_test/results.pkl",
            "driller/inference_/lm_real_driller_test/FlowNet512-1.5AugCosyAAEGray-AggressiveV2-Flat-Pbr-08-driller_lm_real_driller_test_preds.pkl",
        ),
        "duck": (
            "duck/inference_model_final_wo_optim-0bde58bb/lm_real_duck_test/results.pkl",
            "duck/inference_model_final/lm_real_duck_test/FlowNet512-1.5AugCosyAAEGray-AggressiveV2-Flat-Pbr-09-duck-test_lm_real_duck_test_preds.pkl",
        ),
        "eggbox": (
            "eggbox_Rsym/inference_model_final_wo_optim-d0656ca7/lm_real_eggbox_test/results.pkl",
            "eggbox/inference_model_final/lm_real_eggbox_test/FlowNet512-1.5AugCosyAAEGray-AggressiveV2-Flat-Pbr-10-eggbox-test_lm_real_eggbox_test_preds.pkl",
        ),
        "glue": (
            "glue_Rsym/inference_model_final_wo_optim-324d8f16/lm_real_glue_test/results.pkl",
            "glue/inference_model_final/lm_real_glue_test/FlowNet512-1.5AugCosyAAEGray-AggressiveV2-Flat-Pbr-11-glue-test_lm_real_glue_test_preds.pkl",
        ),
        "holepuncher": (
            "holepuncher/inference_model_final_wo_optim-eab19662/lm_real_holepuncher_test/results.pkl",
            "holepuncher/inference_/lm_real_holepuncher_test/FlowNet512-1.5AugCosyAAEGray-AggressiveV2-Flat-Pbr-12-holepuncher_lm_real_holepuncher_test_preds.pkl",
        ),
        "iron": (
            "iron/inference_model_final_wo_optim-025a740e/lm_real_iron_test/results.pkl",
            "iron/inference_model_final/lm_real_iron_test/FlowNet512-1.5AugCosyAAEGray-AggressiveV2-Flat-Pbr-13-iron-test_lm_real_iron_test_preds.pkl",
        ),
        "lamp": (
            "lamp/inference_model_final_wo_optim-34042758/lm_real_lamp_test/results.pkl",
            "lamp/inference_model_final/lm_real_lamp_test/FlowNet512-1.5AugCosyAAEGray-AggressiveV2-Flat-Pbr-14-lamp-test_lm_real_lamp_test_preds.pkl",
        ),
        "phone": (
            "phone/inference_model_final_wo_optim-525a29f8/lm_real_phone_test/results.pkl",
            "phone/inference_/lm_real_phone_test/FlowNet512-1.5AugCosyAAEGray-AggressiveV2-Flat-Pbr-15-phone_lm_real_phone_test_preds.pkl",
        ),
    }

    dset_name = "lm_13_test"
    dprint(dset_name)
    register_datasets([dset_name])

    meta = MetadataCatalog.get(dset_name)
    dprint("MetadataCatalog: ", meta)
    objs = meta.objs

    dset_dicts = DatasetCatalog.get(dset_name)
    scene_im_id_to_gt_index = {d["scene_im_id"]: i for i, d in enumerate(dset_dicts)}

    ren = EGLRenderer(
        model_paths,
        texture_paths=texture_paths,
        vertex_scale=0.001,
        height=IM_H,
        width=IM_W,
        znear=near,
        zfar=far,
        use_cache=True,
    )
    height = IM_H
    width = IM_W
    device = "cuda"
    image_tensor = torch.cuda.FloatTensor(height, width, 4, device=device).detach()
    seg_tensor = torch.cuda.FloatTensor(height, width, 4, device=device).detach()
    pc_cam_tensor = torch.cuda.FloatTensor(height, width, 4, device=device).detach()

    merged_results = {}  # key is scene_im_id, each contains a list of instance dicts

    for obj_name, init_refine_name in init_refine_path_names.items():
        dprint(obj_name)

        total_cnt = 0
        dice_pred_cnt = 0
        dice_gt_cnt = 0
        # if obj_name in ["ape"]:
        #     continue

        init_res_path = osp.join(init_res_root, init_refine_name[0])
        assert osp.exists(init_res_path), init_res_path

        refine_res_path = osp.join(refined_res_root, init_refine_name[1])
        assert osp.exists(refine_res_path), refine_res_path

        init_results = mmcv.load(init_res_path)  # pkl  scene_im_id key, list of preds
        refine_results = mmcv.load(refine_res_path)  # pkl obj_name: im_path: pred_dict  (assume single obj instance)
        cur_refine_results = refine_results[obj_name]
        for im_path, refine_dict in tqdm(cur_refine_results.items()):
            scene_id = int(im_path.split("/")[-3])
            im_id = int(im_path.split("/")[-1].split(".")[0])
            scene_im_id = f"{scene_id}/{im_id}"

            if scene_im_id not in merged_results:
                merged_results[scene_im_id] = []

            if scene_im_id not in scene_im_id_to_gt_index:
                dprint("{} not in gt dicts".format(scene_im_id))
            gt_idx = scene_im_id_to_gt_index[scene_im_id]
            gt_dict = dset_dicts[gt_idx]
            gt_annos = gt_dict["annotations"]

            # assume single precition (for lmo, ycbv, need a loop)
            init_dict = init_results[scene_im_id][0]
            obj_id = init_dict["obj_id"]
            for gt_anno in gt_annos:
                gt_label = gt_anno["category_id"]
                gt_obj = objs[gt_label]
                gt_obj_id = obj2id[gt_obj]
                if obj_id == gt_obj_id:
                    gt_pose = gt_anno["pose"]
                    break

            # NOTE: pose in init_dict should be iter0 pose in refine_dict

            # fetch init pose and refined pose (iter4)
            pose_init = np.zeros((3, 4), dtype=np.float32)
            pose_init[:3, :3] = init_dict["R"]
            pose_init[:3, 3] = init_dict["t"]

            pose_refined = np.zeros((3, 4), dtype=np.float32)
            pose_refined[:3, :3] = refine_dict["iter4"]["R"]
            pose_refined[:3, 3] = refine_dict["iter4"]["t"]

            if "full_mask" in init_dict:
                mask_pred = cocosegm2mask(init_dict["full_mask"], IM_H, IM_W)
            else:
                mask_pred = cocosegm2mask(init_dict["mask"], IM_H, IM_W)

            # get rendered mask from pose_init, pose_refined, gt_pose
            ren_label = obj_ids.index(obj_id)
            ren.render([ren_label], [pose_init], K=K, seg_tensor=seg_tensor, image_tensor=image_tensor)
            ren_mask_init = (seg_tensor[:, :, 0].cpu().numpy() > 0).astype("uint8")
            ren_color_init = image_tensor[:, :, :3].cpu().numpy().astype("uint8")

            ren.render([ren_label], [pose_refined], K=K, seg_tensor=seg_tensor, image_tensor=image_tensor)
            ren_mask_refined = (seg_tensor[:, :, 0].cpu().numpy() > 0).astype("uint8")
            ren_color_refined = image_tensor[:, :, :3].cpu().numpy().astype("uint8")

            ren.render([ren_label], [gt_pose], K=K, seg_tensor=seg_tensor, image_tensor=image_tensor)
            ren_mask_gt = (seg_tensor[:, :, 0].cpu().numpy() > 0).astype("uint8")
            ren_color_gt = image_tensor[:, :, :3].cpu().numpy().astype("uint8")

            dice_pred_init = batch_dice_score(mask_pred[None], ren_mask_init[None])[0]
            dice_pred_refine = batch_dice_score(mask_pred[None], ren_mask_refined[None])[0]
            dice_pred_gt = batch_dice_score(mask_pred[None], ren_mask_gt[None])[0]
            dice_init_gt = batch_dice_score(ren_mask_init[None], ren_mask_gt[None])[0]
            dice_refine_gt = batch_dice_score(ren_mask_refined[None], ren_mask_gt[None])[0]

            re_init_gt, te_init_gt = calc_rt_dist_m(pose_init, gt_pose)
            re_refine_gt, te_refine_gt = calc_rt_dist_m(pose_refined, gt_pose)

            total_cnt += 1
            if dice_pred_refine > dice_pred_init and (re_refine_gt + te_refine_gt) / 2 < (re_init_gt + te_init_gt) / 2:
                dice_pred_cnt += 1
            if dice_refine_gt > dice_init_gt and (re_refine_gt + te_refine_gt) / 2 < (re_init_gt + te_init_gt) / 2:
                dice_gt_cnt += 1

            if VIS:
                font_thickness = 2
                font_scale = 1
                if dice_pred_init > dice_pred_refine:
                    text_color = mmcv.color_val("green")
                    refine_text_color = mmcv.color_val("red")
                else:
                    text_color = mmcv.color_val("red")
                    refine_text_color = mmcv.color_val("green")
                cv2.putText(
                    ren_color_init,
                    f"dice_init_pred: {dice_pred_init * 100:.2f}",
                    (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    text_color,
                    font_thickness,
                )
                cv2.putText(
                    ren_color_refined,
                    f"dice_pred_refine: {dice_pred_refine * 100:.2f}",
                    (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    refine_text_color,
                    font_thickness,
                )
                #####################################
                if dice_init_gt > dice_refine_gt:
                    text_color = mmcv.color_val("green")
                    refine_text_color = mmcv.color_val("red")
                else:
                    text_color = mmcv.color_val("red")
                    refine_text_color = mmcv.color_val("green")
                cv2.putText(
                    ren_color_init,
                    f"dice_init_gt: {dice_init_gt * 100:.2f}",
                    (5, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    text_color,
                    font_thickness,
                )
                cv2.putText(
                    ren_color_refined,
                    f"dice_refine_gt: {dice_refine_gt * 100:.2f}",
                    (5, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    refine_text_color,
                    font_thickness,
                )
                ##########################
                if re_init_gt < re_refine_gt:
                    text_color = mmcv.color_val("green")
                    refine_text_color = mmcv.color_val("red")
                else:
                    text_color = mmcv.color_val("red")
                    refine_text_color = mmcv.color_val("green")
                cv2.putText(
                    ren_color_init,
                    f"re_init_gt: {re_init_gt:.2f}",
                    (5, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    text_color,
                    font_thickness,
                )
                cv2.putText(
                    ren_color_refined,
                    f"re_refine_gt: {re_refine_gt:.2f}",
                    (5, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    refine_text_color,
                    font_thickness,
                )
                ######################################################
                if te_init_gt < te_refine_gt:
                    text_color = mmcv.color_val("green")
                    refine_text_color = mmcv.color_val("red")
                else:
                    text_color = mmcv.color_val("red")
                    refine_text_color = mmcv.color_val("green")
                cv2.putText(
                    ren_color_init,
                    f"te_init_gt: {te_init_gt*100:.2f}cm",
                    (5, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    text_color,
                    font_thickness,
                )
                cv2.putText(
                    ren_color_refined,
                    f"te_refine_gt: {te_refine_gt*100:.2f}cm",
                    (5, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    refine_text_color,
                    font_thickness,
                )

                color = mmcv.imread(im_path, "color")
                show_ims = [
                    color[:, :, ::-1],
                    ren_color_gt[:, :, ::-1],
                    ren_color_init[:, :, ::-1],
                    ren_color_refined[:, :, ::-1],
                ]
                show_titles = ["color", "ren_color_gt", "ren_color_init", "ren_color_refined"]
                grid_show(show_ims, show_titles, row=2, col=2)

            # cur_merged = {
            #     "pose_est":
            # }
        dprint(f"{obj_name} dice_pred_cnt: {dice_pred_cnt}, dice_gt_cnt: {dice_gt_cnt}, total: {total_cnt}")
