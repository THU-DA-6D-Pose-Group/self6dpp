import os.path as osp
import sys
import numpy as np
import mmcv
from tqdm import tqdm
from functools import cmp_to_key

cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../../../../"))
sys.path.insert(0, PROJ_ROOT)
# from lib.pysixd import inout
from lib.utils.bbox_utils import xyxy_to_xywh
from lib.utils.utils import iprint, wprint

res_path = osp.join(PROJ_ROOT, "output/Cosypose/results.pkl")

if __name__ == "__main__":
    new_res_path = osp.join(
        PROJ_ROOT,
        "datasets/BOP_DATASETS/ycbv/test/init_poses/",
        "resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_GdrnPose_wBboxCrop_wCosyPose_ycbvTrainRealUw.json",
    )
    if osp.exists(new_res_path):
        wprint("{} already exists! overriding!".format(new_res_path))

    results = mmcv.load(res_path)["gdrn_init/refiner/iteration=2"]
    refined_poses = results.poses  # N*4*4 tensor
    init_poses = results.poses_input
    obj_strs = results.infos.label  # str f'obj_{:06d}'
    scene_ids = results.infos.scene_id
    view_ids = results.infos.view_id
    scores = results.infos.score
    bboxes_crop = results.boxes_crop  # xyxy, N*4 tensor

    new_res_dict = {}
    for i in tqdm(range(len(results))):
        pose_refine = np.array(refined_poses[i])[:3]
        pose_est = np.array(init_poses[i])[:3]
        scene_id = scene_ids[i]
        view_id = view_ids[i]
        scene_im_id = f"{scene_id}/{view_id}"
        score = scores[i]
        obj_id = int(obj_strs[i][-6:])
        bbox_crop = np.array(bboxes_crop[i])
        bbox_crop_xywh = xyxy_to_xywh(bbox_crop)

        cur_new_res = {
            "obj_id": obj_id,
            "score": float(score),
            # "bbox_crop": bbox_crop_xywh.tolist(),
            "bbox_est": bbox_crop_xywh.tolist(),
            "pose_est": pose_est.tolist(),
            "pose_refine": pose_refine.tolist(),
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
    # inout.save_json(new_res_path, new_res_dict_sorted)
    mmcv.dump(new_res_dict_sorted, new_res_path)
    iprint()
    iprint("new result path: {}".format(new_res_path))
