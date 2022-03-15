"""Generate 50k egl rendered images, each contains 3-10 objects, with random
lighting, random background, random pose."""
import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
import random
import math
import numpy as np
import mmcv
import torch
import os.path as osp
import sys
from tqdm import tqdm
import setproctitle

cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../../../../"))
sys.path.insert(0, PROJ_ROOT)
from lib.pysixd import inout, misc, view_sampler, transform
from lib.utils.utils import dprint
from lib.egl_renderer.egl_renderer_v3 import EGLRenderer
from lib.utils.mask_utils import binary_mask_to_rle, mask2bbox_xywh
from lib.utils.bbox_utils import xywh_to_xyxy
from lib.vis_utils.image import vis_image_mask_bbox_cv2, grid_show
import ref


seed = 1
random.seed(seed)
np.random.seed(seed)

setproctitle.setproctitle(osp.basename(__file__).split(".")[0])

ref_key = "lm_full"
data_ref = ref.__dict__[ref_key]

data_root = osp.join(PROJ_ROOT, "datasets/BOP_DATASETS/lm")
model_dir = osp.join(data_root, "models")

bg_root = osp.join(PROJ_ROOT, "datasets/VOCdevkit/VOC2012/")
bg_paths = [
    osp.join(bg_root, "JPEGImages", fn.name) for fn in os.scandir(osp.join(bg_root, "JPEGImages")) if ".jpg" in fn.name
]


result_root = osp.join(data_root, "train_egl")
mmcv.mkdir_or_exist(result_root)

VERTEX_SCALE = 0.001
DEPTH_FACTOR = 10000
K = data_ref.camera_matrix
IM_H = 480
IM_W = 640

MIN_NUM_OBJ = 3
MAX_NUM_OBJ = 10
RANDOM_TRANS_MIN = [-0.35, -0.35, 0.5]
RANDOM_TRANS_MAX = [0.35, 0.35, 1.3]
N_TOTAL = 50000  # 50k

"""
MIN_N_VIEWS = 2536  # AAE  # min(max(N_TOTAL, 3000), 30000)
RADIUS = 1.0
AZIMUTH_RANGE = (0, 2 * math.pi)
ELEV_RANGE = (-0.5 * math.pi, 0.5 * math.pi)
# for each view generate a number of in-plane rotations to cover full SO(3)
NUM_CYCLO = 36

# list of {"R": , "t": }
all_views, _ = view_sampler.sample_views(MIN_N_VIEWS, RADIUS, AZIMUTH_RANGE, ELEV_RANGE)
Rs = []
for view in all_views:
    for cyclo in np.linspace(0, 2.*np.pi, NUM_CYCLO):
        rot_z = np.array([[np.cos(-cyclo), -np.sin(-cyclo), 0], [np.sin(-cyclo), np.cos(-cyclo), 0], [0, 0, 1]])
        Rs.append(rot_z.dot(view['R']))
Rs = np.array(Rs, dtype="float32")  # 2536*36 = 91296
"""

obj_ids = list(data_ref.id2obj.keys())
models = [inout.load_ply(model_path, vertex_scale=VERTEX_SCALE) for model_path in data_ref.model_paths]

tensor_kwargs = {"device": torch.device("cuda"), "dtype": torch.float32}
image_tensor = torch.empty((IM_H, IM_W, 4), **tensor_kwargs).detach()
seg_tensor = torch.empty((IM_H, IM_W, 4), **tensor_kwargs).detach()
pc_cam_tensor = torch.empty((IM_H, IM_W, 4), **tensor_kwargs).detach()

ren = EGLRenderer(
    model_paths=data_ref.model_paths,
    texture_paths=data_ref.texture_paths,
    vertex_scale=VERTEX_SCALE,
    K=K,
    height=IM_H,
    width=IM_W,
    use_cache=True,
)

VIS = False

if __name__ == "__main__":
    gts = {}
    for i_im in tqdm(range(N_TOTAL)):
        if MIN_NUM_OBJ == MAX_NUM_OBJ:
            num_obj = MIN_NUM_OBJ
        else:
            num_obj = np.random.randint(MIN_NUM_OBJ, MAX_NUM_OBJ)

        cur_labels = np.random.choice(len(obj_ids), num_obj).tolist()
        cur_obj_names = [data_ref.objects[_l] for _l in cur_labels]

        cur_Rs = []
        cur_ts = []
        cur_ts_norm = []
        for inst_i, obj_id in enumerate(cur_labels):
            success = False
            while not success:
                t_min = RANDOM_TRANS_MIN
                t_max = RANDOM_TRANS_MAX
                t = np.array([random.uniform(_min, _max) for _min, _max in zip(t_min, t_max)])
                R = transform.random_rotation_matrix()[:3, :3]
                t_norm = t / np.linalg.norm(t)

                t_im = np.dot(K, t.reshape(3, 1))
                t_im_cx = t_im[0] / t_im[2]
                t_im_cy = t_im[1] / t_im[2]
                if t_im_cx >= IM_W - 20 or t_im_cx <= 20 or t_im_cy >= IM_H - 20 or t_im_cy <= 20:
                    continue

                if len(cur_ts_norm) > 0 and np.any(np.dot(np.array(cur_ts_norm), t_norm.reshape(3, 1)) > 0.99):
                    success = False
                    # dprint(f"fail: {i_im} {inst_i}")
                else:
                    cur_ts_norm.append(t_norm)
                    cur_ts.append(t)
                    cur_Rs.append(R)
                    success = True
        cur_poses = [np.hstack([_R, _t.reshape(3, 1)]) for _R, _t in zip(cur_Rs, cur_ts)]
        # apply random lighting
        light_pos = np.random.uniform(-1, 1, 3)
        intensity = np.random.uniform(0.8, 2)
        light_color = intensity * np.random.uniform(0.9, 1.1, 3)
        ren.render(
            cur_labels,
            cur_poses,
            K=K,
            light_pos=light_pos,
            light_color=light_color,
            to_bgr=True,
            to_255=True,
            image_tensor=image_tensor,
            pc_cam_tensor=pc_cam_tensor,
            seg_tensor=seg_tensor,
        )

        color_gl = image_tensor[:, :, :3].cpu().numpy().astype("uint8")
        depth_gl = (pc_cam_tensor[:, :, 2].cpu().numpy() * DEPTH_FACTOR).astype("uint16")
        fg_mask = (seg_tensor[:, :, 0].cpu().numpy() > 0).astype(np.bool)

        # random bg
        bg_ind = random.randint(0, len(bg_paths) - 1)
        bg_filename = bg_paths[bg_ind]
        bg_img = mmcv.imread(bg_filename, "color")
        bg_img = mmcv.imresize(bg_img, size=(IM_W, IM_H))
        mask_bg = ~fg_mask
        color_gl[mask_bg] = bg_img[mask_bg]
        color_gl = color_gl.astype(np.uint8)

        # instance labels
        full_masks = []  # full mask
        bboxes = []
        for label_i, pose_i in zip(cur_labels, cur_poses):
            ren.render([label_i], [pose_i], K=K, seg_tensor=seg_tensor)
            mask_np = (seg_tensor[:, :, 0].cpu().numpy() > 0).astype("uint8")
            full_masks.append(mask_np)
            bboxes.append(np.array(mask2bbox_xywh(mask_np)))

        full_rles = [binary_mask_to_rle(_mask) for _mask in full_masks]

        # get visib mask
        tz_list = [_p[2, 3] for _p in cur_poses]
        dist_inds = np.argsort(tz_list)[::-1]  # descending order
        scene_mask_with_inst_labels = np.zeros((IM_H, IM_W), dtype="uint8")
        # paste instance masks, with farther objects first
        for i_inst, i_dist in enumerate(dist_inds):
            scene_mask_with_inst_labels[full_masks[i_dist] > 0] = i_dist + 1

        results = []
        visib_masks = []
        for inst_i, label_i in enumerate(cur_labels):
            obj_id = obj_ids[label_i]
            cur_visib_mask = (scene_mask_with_inst_labels == (1 + inst_i)).astype("uint8")
            visib_masks.append(cur_visib_mask)
            inst = {
                "obj_id": obj_id,
                "bbox": bboxes[inst_i].tolist(),
                "pose": cur_poses[inst_i].tolist(),
                "mask_full": full_rles[inst_i],
                "mask_visib": binary_mask_to_rle(cur_visib_mask),
            }
            results.append(inst)
        gts[str(i_im)] = results

        # import ipdb; ipdb.set_trace()

        if VIS:
            bboxes_xyxy = xywh_to_xyxy(np.array(bboxes))
            im_vis_full_mask = vis_image_mask_bbox_cv2(color_gl, full_masks, bboxes_xyxy, labels=cur_obj_names)
            im_vis_visib_mask = vis_image_mask_bbox_cv2(color_gl, visib_masks, bboxes_xyxy, labels=cur_obj_names)
            show_ims = [
                color_gl[:, :, ::-1],
                im_vis_full_mask[:, :, ::-1],
                im_vis_visib_mask[:, :, ::-1],
                depth_gl / DEPTH_FACTOR,
            ]
            show_titles = [
                "color_gl",
                "im_vis_full_mask",
                "im_vis_visib_mask",
                "depth_gl",
            ]
            grid_show(show_ims, show_titles, row=1, col=4)

        # save cur results
        im_path = osp.join(result_root, f"rgb/{i_im:06d}.jpg")
        depth_path = osp.join(result_root, f"depth/{i_im:06d}.png")
        label_path = osp.join(result_root, f"label/{i_im:06d}.json")
        mmcv.imwrite(color_gl, im_path)
        mmcv.imwrite(depth_gl, depth_path)
        mmcv.mkdir_or_exist(osp.dirname(label_path))
        inout.save_json(label_path, results)

    # done, save all gts into one
    all_gt_path = osp.join(result_root, "gt.json")
    inout.save_json(all_gt_path, gts)
