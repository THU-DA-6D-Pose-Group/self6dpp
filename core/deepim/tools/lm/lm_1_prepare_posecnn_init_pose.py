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

data_root = osp.join(PROJ_ROOT, "datasets/BOP_DATASETS/lm")
test_root = osp.join(data_root, "test")
image_set_dir = osp.join(data_root, "image_set")

# original posecnn results
posecnn_results_dir = osp.join(test_root, "PoseCNN_LINEMOD_6D_results")

# our format
init_pose_dir = osp.join(test_root, "init_poses")
mmcv.mkdir_or_exist(init_pose_dir)
init_pose_path = osp.join(init_pose_dir, "init_pose_posecnn_lm.json")

idx2class = {
    1: "ape",
    2: "benchvise",
    # 3: 'bowl',
    4: "camera",
    5: "can",
    6: "cat",
    # 7: 'cup',
    8: "driller",
    9: "duck",
    10: "eggbox",
    11: "glue",
    12: "holepuncher",
    13: "iron",
    14: "lamp",
    15: "phone",
}
classes = idx2class.values()
classes = sorted(classes)

K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])


if __name__ == "__main__":
    results = {}
    num_not_exist = 0
    num_not_found = 0
    num_total = 0
    for obj_id, obj_name in tqdm(idx2class.items()):
        scene_id = obj_id  # for lm, scene_id is obj_id
        model = inout.load_ply(
            osp.join(data_root, f"models/obj_{obj_id:06d}.ply"),
            vertex_scale=0.001,
        )

        idx_file = osp.join(image_set_dir, f"{obj_name}_test.txt")
        with open(idx_file, "r") as f:
            indices = [line.strip("\r\n") for line in f]
        for i_res, im_idx in enumerate(tqdm(indices)):
            num_total += 1
            posecnn_obj_name = obj_name if obj_name != "benchvise" else "benchviseblue"
            posecnn_result_path = osp.join(posecnn_results_dir, posecnn_obj_name, f"{i_res:04d}.mat")
            if not osp.exists(posecnn_result_path):
                print(f"not result file: {posecnn_result_path}")
                num_not_exist += 1
                continue

            posecnn_res = loadmat(posecnn_result_path)
            found = posecnn_res["rois"][:, 1]
            if found == -1:  # not detected
                print(f"not detected: {obj_name} {im_idx}")
                num_not_found += 1
                continue
            proposal_idx = np.where(found == 1)
            assert len(proposal_idx) == 1
            pose_q = posecnn_res["poses"][proposal_idx].reshape(7)
            pose_m = se3_q2m(pose_q)

            bbox_from_pose = misc.compute_2d_bbox_xywh_from_pose(
                model["pts"], pose_m, K, width=640, height=480, clip=True
            )

            int_im_idx = int(im_idx)
            scene_im_id = f"{scene_id}/{int_im_idx}"
            results[scene_im_id] = []
            cur_res = {
                "obj_id": obj_id,
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
