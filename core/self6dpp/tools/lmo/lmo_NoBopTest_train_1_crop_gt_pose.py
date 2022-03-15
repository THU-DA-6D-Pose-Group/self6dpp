import os
import random

os.environ["PYOPENGL_PLATFORM"] = "egl"
import os.path as osp
import sys
import numpy as np
import mmcv
import cv2
import time
import torch
import matplotlib
from matplotlib import pyplot as plt
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../../../../"))
sys.path.insert(0, PROJ_ROOT)
from lib.egl_renderer.egl_renderer_v3 import EGLRenderer
from lib.pysixd import inout, misc
from core.utils.my_visualizer import MyVisualizer, _GREY, _GREEN, _BLUE, _RED
from lib.vis_utils.image import grid_show, heatmap, vis_image_bboxes_cv2, vis_image_mask_bbox_cv2
from lib.vis_utils.cmap_plt2cv import colorize
from lib.utils.mask_utils import cocosegm2mask, batch_dice_score
from lib.utils.bbox_utils import xywh_to_xyxy
from lib.utils.mask_utils import get_contour_cv2, get_edge
from lib.utils.utils import dprint, iprint, wprint
from lib.pysixd.pose_error import calc_rt_dist_m
from core.gdrn_modeling.datasets.dataset_factory import register_datasets
import ref
from core.utils.data_utils import crop_resize_by_warp_affine


DEBUG = False

dpi = matplotlib.rcParams["figure.dpi"]
iprint("dpi", dpi)

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
NEAR = 0.01
FAR = 6.5
K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])
DEPTH_FACTOR = 1000.0
DEPTH_MIN = 0.001
DEPTH_MAX = 1.6

BOX_GAP = 10
CROP_SIZE = 256
DZI_PAD_SCALE = 1.5

data_root = osp.normpath(osp.join(PROJ_ROOT, "datasets/BOP_DATASETS/lmo/test"))
model_dir = osp.normpath(osp.join(PROJ_ROOT, "datasets/BOP_DATASETS/lmo/models"))
obj_ids = [_id for _id in id2obj]
model_paths = [osp.join(model_dir, f"obj_{cls_idx:06d}.ply") for cls_idx in id2obj]
texture_paths = None

models = [inout.load_ply(m_path, vertex_scale=0.001) for m_path in model_paths]
kpts_3d_list = [misc.get_bbox3d_and_center(_model["pts"]) for _model in models]


if __name__ == "__main__":

    dset_name = "lmo_NoBopTest_driller_train"
    iprint(dset_name)
    register_datasets([dset_name])

    meta = MetadataCatalog.get(dset_name)
    iprint("MetadataCatalog: ", meta)
    objs = meta.objs

    dset_dicts = DatasetCatalog.get(dset_name)
    scene_im_id_to_gt_index = {d["scene_im_id"]: i for i, d in enumerate(dset_dicts)}

    ren = EGLRenderer(
        model_paths,
        texture_paths=texture_paths,
        vertex_scale=0.001,
        height=IM_H,
        width=IM_W,
        znear=NEAR,
        zfar=FAR,
        use_cache=True,
    )
    height = IM_H
    width = IM_W
    device = "cuda"
    image_tensor = torch.cuda.FloatTensor(height, width, 4, device=device).detach()
    seg_tensor = torch.cuda.FloatTensor(height, width, 4, device=device).detach()

    vis_save_root = osp.join(PROJ_ROOT, "output/vis/{}".format(dset_name))
    print("vis_save_root", vis_save_root)
    mmcv.mkdir_or_exist(vis_save_root)

    def _rle2mask(rle):
        return cocosegm2mask(rle, IM_H, IM_W)

    def process_single(dic_id):
        gt_dict = dset_dicts[dic_id]
        gt_annos = gt_dict["annotations"]
        scene_im_id = gt_dict["scene_im_id"]
        iprint(scene_im_id)

        # load original image and depth ----------------------------------
        scene_id = int(scene_im_id.split("/")[0])
        im_id = int(scene_im_id.split("/")[1])
        im_path = osp.join(data_root, f"{scene_id:06d}/rgb/{im_id:06d}.png")
        depth_path = osp.join(data_root, f"{scene_id:06d}/depth/{im_id:06d}.png")
        im = mmcv.imread(im_path, "color")
        depth = mmcv.imread(depth_path, "unchanged") / DEPTH_FACTOR

        vis_dict = {
            "im": im[:, :, ::-1],
            "depth": depth,
        }
        im_show_titles = [_k for _k, _v in vis_dict.items()]
        im_show_ims = [_v for _k, _v in vis_dict.items()]
        for anno_i, gt_anno in enumerate(gt_annos):
            gt_label = gt_anno["category_id"]
            obj_name = objs[gt_label]
            obj_id = obj2id[obj_name]

            ren_label = obj_ids.index(obj_id)
            obj_name = id2obj[obj_id]

            bbox_gt = gt_anno["bbox"]
            bbox_gt_xyxy = np.array(BoxMode.convert(bbox_gt, gt_anno["bbox_mode"], BoxMode.XYXY_ABS))

            gt_pose = gt_anno["pose"]
            gt_mask_vis = _rle2mask(gt_anno["segmentation"])
            if "mask_full" in gt_anno:
                gt_mask_full = _rle2mask(gt_anno["mask_full"])

            # get crop bbox via projected 3d bbox
            kpt3d = kpts_3d_list[ren_label]
            kpt2d_gt = misc.project_pts(kpt3d, K, gt_pose[:3, :3], gt_pose[:3, 3])

            # get rendered mask
            ren.render([ren_label], [gt_pose], K=K, seg_tensor=seg_tensor)
            ren_mask_gt = seg_tensor[:, :, 0].cpu().numpy().astype("bool").astype("uint8")

            maxx, maxy, minx, miny = 0, 0, 1000, 1000
            for i in range(len(kpt2d_gt)):
                maxx, maxy, minx, miny = (
                    max(maxx, kpt2d_gt[i][0]),
                    max(maxy, kpt2d_gt[i][1]),
                    min(minx, kpt2d_gt[i][0]),
                    min(miny, kpt2d_gt[i][1]),
                )
            center_common = np.array([(minx + maxx) / 2, (miny + maxy) / 2])
            scale_common = max(maxx - minx, maxy - miny) + BOX_GAP  # * 3  # + 10

            # get zoomed bbox
            bbox_gt_xyxy_zoom = bbox_gt_xyxy.copy()
            bbox_gt_xyxy_zoom[0] = (bbox_gt_xyxy[0] - (center_common[0] - scale_common / 2)) * CROP_SIZE / scale_common
            bbox_gt_xyxy_zoom[1] = (bbox_gt_xyxy[1] - (center_common[1] - scale_common / 2)) * CROP_SIZE / scale_common
            bbox_gt_xyxy_zoom[2] = (bbox_gt_xyxy[2] - (center_common[0] - scale_common / 2)) * CROP_SIZE / scale_common
            bbox_gt_xyxy_zoom[3] = (bbox_gt_xyxy[3] - (center_common[1] - scale_common / 2)) * CROP_SIZE / scale_common

            # get zoomed projects 3d bboxes
            kpt2d_gt_zoom = kpt2d_gt.copy()
            for i in range(len(kpt2d_gt)):
                # gt
                kpt2d_gt_zoom[i][0] = (
                    (kpt2d_gt[i][0] - (center_common[0] - scale_common / 2)) * CROP_SIZE / scale_common
                )
                kpt2d_gt_zoom[i][1] = (
                    (kpt2d_gt[i][1] - (center_common[1] - scale_common / 2)) * CROP_SIZE / scale_common
                )

            # crop resize (zoom)
            zoomed_im_common = crop_resize_by_warp_affine(im, center_common, scale_common, CROP_SIZE)
            vis_dict[f"{obj_name}_zoomed_im_common"] = zoomed_im_common[:, :, ::-1]

            zoomed_ren_mask_gt_common = crop_resize_by_warp_affine(ren_mask_gt, center_common, scale_common, CROP_SIZE)
            edge_w = 4
            zoomed_edge_gt_common = get_edge(zoomed_ren_mask_gt_common, edge_w, out_channel=3)
            zoomed_edge_gt_common[:, :, [0, 1]] *= -1  # blue

            # cropped im with bbox_gt
            zoomed_im_common_bbox_gt = vis_image_bboxes_cv2(
                zoomed_im_common, [bbox_gt_xyxy_zoom], box_color=(200, 200, 10)
            )  # cyan
            vis_dict[f"{obj_name}_zoomed_im_common_bbox_gt"] = zoomed_im_common_bbox_gt[:, :, ::-1]

            # cropped im with projected 3d bboxes -----------------------------------------------
            linewidth = 3

            # gt pose --------------------
            visualizer = MyVisualizer(zoomed_im_common[:, :, ::-1], meta)
            visualizer.draw_bbox3d_and_center(
                kpt2d_gt_zoom, top_color=_BLUE, bottom_color=_GREY, linewidth=linewidth, draw_center=True
            )
            zoomed_im_common_gt_pose = visualizer.get_output().get_image()
            vis_dict[f"{obj_name}_zoomed_im_common_gt_pose"] = zoomed_im_common_gt_pose

            # gt pose's edge -------
            zoomed_im_common_gt_edge = zoomed_im_common.copy()[:, :, ::-1]  # rgb
            zoomed_im_common_gt_edge[zoomed_edge_gt_common > 0] = 255
            zoomed_im_common_gt_edge[zoomed_edge_gt_common < 0] = 0
            vis_dict[f"{obj_name}_zoomed_im_common_gt_edge"] = zoomed_im_common_gt_edge

            # ----------------------------------------------------------------------
            zoomed_depth_common = crop_resize_by_warp_affine(depth, center_common, scale_common, CROP_SIZE)
            zoomed_depth_common[0, 0] = 2.1  # HACK: make depths looks similar
            vis_dict[f"{obj_name}_zoomed_depth_common"] = zoomed_depth_common

            # zoomed masks
            mask_interp = cv2.INTER_NEAREST

            zoomed_gt_mask_vis = crop_resize_by_warp_affine(
                gt_mask_vis, center_common, scale_common, CROP_SIZE, interpolation=mask_interp
            )
            zoomed_gt_mask_full = crop_resize_by_warp_affine(
                gt_mask_full, center_common, scale_common, CROP_SIZE, interpolation=mask_interp
            )
            vis_dict[f"{obj_name}_zoomed_gt_mask_vis"] = zoomed_gt_mask_vis
            vis_dict[f"{obj_name}_zoomed_gt_mask_full"] = zoomed_gt_mask_full

            # vis ------------------------------------------
            if DEBUG:
                show_titles = im_show_titles + [_k for _k, _v in vis_dict.items() if obj_name in _k]
                show_ims = im_show_ims + [_v for _k, _v in vis_dict.items() if obj_name in _k]
                ncol = 4
                nrow = int(np.ceil(len(show_ims) / ncol))
                nrow_per_fig = 4
                n_fig = int(np.ceil(nrow / nrow_per_fig))
                for fig_i in range(n_fig):
                    fig_start = ncol * nrow_per_fig * fig_i
                    fig_end = ncol * nrow_per_fig * (fig_i + 1)
                    # if fig_i == n_fig - 1:
                    #     cur_nrow = nrow % nrow_per_fig
                    # else:
                    #     cur_nrow = nrow_per_fig
                    cur_nrow = nrow_per_fig
                    # NOTE: grid show's line2D has artifacts due to upsampling to a larger dpi! need to save individual images
                    grid_show(
                        show_ims[fig_start:fig_end], show_titles[fig_start:fig_end], row=cur_nrow, col=ncol, show=False
                    )
                plt.show()

        if True:  # save individual images
            cur_save_dir = osp.join(vis_save_root, f"{scene_id:06d}/{im_id:06d}")
            mmcv.mkdir_or_exist(cur_save_dir)
            for show_title, show_im in vis_dict.items():
                cur_save_path = osp.join(cur_save_dir, f"{show_title}.png")
                if "pose" in show_title:
                    # NOTE: no big difference
                    cv2.imwrite(cur_save_path, show_im[:, :, ::-1])
                else:
                    cur_h, cur_w = show_im.shape[:2]
                    if len(show_im.shape) == 2:  # save both viridis and gray versions
                        if "mask" in show_title:
                            vmin = 0
                        else:
                            vmin = None
                        cv2.imwrite(cur_save_path, colorize(show_im, "viridis", vmin=vmin))

                        cur_gray_save_path = osp.join(cur_save_dir, f"{show_title}_gray.png")
                        cv2.imwrite(cur_gray_save_path, colorize(show_im, "gray", vmin=vmin))
                    else:
                        cv2.imwrite(cur_save_path, show_im[:, :, ::-1])

    # -------------------------------------------
    all_ids = [i for i in range(len(dset_dicts))]
    tic = time.perf_counter()
    for sel_id in tqdm(all_ids):
        process_single(sel_id)
    total_time = time.perf_counter() - tic
    iprint("total time: ", total_time)
