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
    gdrn_res_path = osp.join(
        PROJ_ROOT,
        "datasets/BOP_DATASETS/ycbv/test/init_poses/",
        "resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_ycbvTrainRealUw_GdrnPose_withYolov4PbrBbox.json",
    )
    new_res_path = osp.join(
        PROJ_ROOT,
        "datasets/BOP_DATASETS/ycbv/test/init_poses/",
        "resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_GdrnPose_wBboxCrop_wCosyPose_ycbvTrainRealUw.json",
    )
    if osp.exists(new_res_path):
        wprint("{} already exists! overriding!".format(new_res_path))

    bases = mmcv.load(gdrn_res_path)
    results = mmcv.load(res_path)["gdrn_init/refiner/iteration=2"]

    def filter_predictions_csv(preds, scene_id, im_id):
        mask = preds.infos["scene_id"] == scene_id
        mask = np.logical_and(mask, preds.infos["view_id"] == im_id)
        # mask = np.logical_and(mask, preds.infos['label'] == f'obj_{obj_id:06d}')

        keep_ids = np.where(mask)[0]
        pred = preds[keep_ids]
        return pred

    new_res_dict = {}
    for scene_im_id, infos in tqdm(bases.items()):
        scene_id, im_id = scene_im_id.split("/")
        scene_id = int(scene_id)
        im_id = int(im_id)
        results_per_im = filter_predictions_csv(results, scene_id, im_id)
        for i, res in enumerate(infos):
            obj_id = res["obj_id"]
            mask = results_per_im.infos["label"] == f"obj_{obj_id:06d}"
            keep_ids = np.where(mask)[0]
            if len(keep_ids) == 0:
                iprint("Refined Pose NOT Found!")
                continue
            if len(keep_ids) > 1:
                iprint("More than one refined poses are found, pick one by score")

            # sorted according to score by default
            pose_refine = results_per_im[keep_ids[0]].poses  # 4*4 tensor
            pose_refine = np.array(pose_refine)[:3]

            res.update({"pose_refine": pose_refine.tolist()})
            if scene_im_id not in new_res_dict:
                new_res_dict[scene_im_id] = []
            new_res_dict[scene_im_id].append(res)

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
