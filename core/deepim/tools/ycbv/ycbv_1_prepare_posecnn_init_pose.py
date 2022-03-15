import numpy as np
import mmcv
from scipy.io import loadmat
import os.path as osp
import sys
from tqdm import tqdm
import setproctitle

cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../../../../"))
sys.path.insert(0, PROJ_ROOT)
from lib.pysixd import inout, misc
from lib.pysixd.RT_transform import se3_q2m

setproctitle.setproctitle(osp.basename(__file__).split(".")[0])

data_root = osp.join(PROJ_ROOT, "datasets/BOP_DATASETS/ycbv")
test_root = osp.join(data_root, "test")
image_set_dir = osp.join(data_root, "image_sets")

# original posecnn results
posecnn_results_dir = osp.join(test_root, "results_PoseCNN_RSS2018")

# our format
init_pose_dir = osp.join(test_root, "init_poses")
mmcv.mkdir_or_exist(init_pose_dir)
init_pose_path = osp.join(init_pose_dir, "init_pose_posecnn_rss18_ycbv.json")

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
objects = sorted(id2obj.values())


K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])


if __name__ == "__main__":
    results = {}

    idx_file = osp.join(image_set_dir, "keyframe.txt")
    with open(idx_file, "r") as f:
        indices = [line.strip("\r\n") for line in f]

    models = {}
    for obj_id in id2obj:
        models[obj_id] = inout.load_ply(
            osp.join(data_root, f"models/obj_{obj_id:06d}.ply"),
            vertex_scale=0.001,
        )

    num_not_exist = 0
    num_not_found = 0
    num_total = 0

    for i_res, scene_im_idx in enumerate(tqdm(indices)):
        scene_im_split = scene_im_idx.split("/")
        scene_id = int(scene_im_split[0])
        int_im_idx = int(scene_im_split[1])

        num_total += 1
        posecnn_result_path = osp.join(posecnn_results_dir, f"{i_res:06d}.mat")

        if not osp.exists(posecnn_result_path):
            print(f"not result file: {posecnn_result_path}")
            num_not_exist += 1
            continue

        posecnn_res = loadmat(posecnn_result_path)
        pred_obj_ids = posecnn_res["rois"][:, 1]
        if len(pred_obj_ids) < 1:
            print(f"not detected: {scene_im_idx}")
            num_not_found += 1
            continue

        scene_im_id = f"{scene_id}/{int_im_idx}"
        results[scene_im_id] = []
        for pred_i, pred_obj_id in enumerate(pred_obj_ids):
            pred_obj_id = int(pred_obj_id)
            pose_q = posecnn_res["poses"][pred_i]
            pose_m = se3_q2m(pose_q)

            model = models[pred_obj_id]
            bbox_from_pose = misc.compute_2d_bbox_xywh_from_pose(
                model["pts"], pose_m, K, width=640, height=480, clip=True
            )

            cur_res = {
                "obj_id": pred_obj_id,
                "pose_est": pose_m.tolist(),
                "bbox_est": bbox_from_pose.tolist(),
                "score": 1.0,
            }
            results[scene_im_id].append(cur_res)

    print(init_pose_path)
    print(f"num not exist: {num_not_exist}/{num_total}={num_not_exist/num_total*100:.2f}%")
    print(f"num not found: {num_not_found}/{num_total}={num_not_found/num_total*100:.2f}%")
    """
    num not exist: 0
    num not found: 0
    """
    inout.save_json(init_pose_path, results, sort=False)
