# -*- coding: utf-8 -*-
"""inference on dataset; save results; evaluate with bop_toolkit (if gt is
available)"""
import datetime
import itertools
import logging
import os.path as osp
import random
import time
from collections import OrderedDict

import cv2
import mmcv
import numpy as np
import ref
import torch
from torch.cuda.amp import autocast
from transforms3d.quaternions import quat2mat
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import DatasetEvaluator, DatasetEvaluators, inference_context
from detectron2.utils.logger import log_every_n_seconds, log_first_n

from lib.pysixd import inout, misc
from lib.pysixd.pose_error import te
from lib.utils.mask_utils import binary_mask_to_rle
from lib.vis_utils.image import grid_show, vis_image_bboxes_cv2

from core.deepim.engine.engine_utils import get_out_coor, get_out_mask
from core.deepim.engine.batching import batch_data, batch_updater
from core.deepim.engine.test_utils import eval_cached_results, save_and_eval_results, to_list
from core.utils.my_comm import all_gather, get_world_size, is_main_process, synchronize


class DeepIM_Evaluator(DatasetEvaluator):
    """use bop toolkit to evaluate."""

    def __init__(self, cfg, dataset_name, distributed, output_dir, train_objs=None, renderer=None):
        self.cfg = cfg
        self.n_iter_test = cfg.MODEL.DEEPIM.N_ITER_TEST

        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        # if test objs are just a subset of train objs
        self.train_objs = train_objs
        self.renderer = renderer

        self._metadata = MetadataCatalog.get(dataset_name)
        self.data_ref = ref.__dict__[self._metadata.ref_key]
        self.obj_names = self._metadata.objs
        self.obj_ids = [self.data_ref.obj2id[obj_name] for obj_name in self.obj_names]
        # with contextlib.redirect_stdout(io.StringIO()):
        #     self._coco_api = COCO(self._metadata.json_file)
        self.model_paths = [
            osp.join(self.data_ref.model_dir, "obj_{:06d}.ply".format(obj_id)) for obj_id in self.obj_ids
        ]
        self.models_3d = [
            inout.load_ply(model_path, vertex_scale=self.data_ref.vertex_scale) for model_path in self.model_paths
        ]

        # eval cached
        if cfg.VAL.EVAL_CACHED or cfg.VAL.EVAL_PRINT_ONLY:
            eval_cached_results(self.cfg, self._output_dir, obj_ids=self.obj_ids)

    def reset(self):
        self._predictions = []

    def _maybe_adapt_label_cls_name(self, label):
        """convert label in test dataset to label in train dataset; if not in
        train, return None."""
        if self.train_objs is not None:
            cls_name = self.obj_names[label]
            if cls_name not in self.train_objs:
                return None, None  # this class was not trained
            label = self.train_objs.index(cls_name)
        else:
            cls_name = self.obj_names[label]
        return label, cls_name

    def get_fps_and_center(self, pts, num_fps=8, init_center=True):
        from core.csrc.fps.fps_utils import farthest_point_sampling

        avgx = np.average(pts[:, 0])
        avgy = np.average(pts[:, 1])
        avgz = np.average(pts[:, 2])
        fps_pts = farthest_point_sampling(pts, num_fps, init_center=init_center)
        res_pts = np.concatenate([fps_pts, np.array([[avgx, avgy, avgz]])], axis=0)
        return res_pts

    def process(self, inputs, batch, outputs, out_dict):
        """
        Args:
            inputs: the batch from data loader
                It is a list of dict. Each dict corresponds to an image and
                    contains keys like "height", "width", "file_name", "scene_im_id".
            batch: the batch (maybe flattened) to the model
            outputs: stores time
            out_dict: the predictions of the model
        """
        pose_est_dict = {}
        for _i in range(self.n_iter_test + 1):
            pose_est_dict[f"iter{_i}"] = out_dict[f"pose_{_i}"].detach().cpu().numpy()
        batch_im_ids = batch["im_id"].detach().cpu().numpy().tolist()
        batch_inst_ids = batch["inst_id"].detach().cpu().numpy().tolist()
        batch_labels = batch["obj_cls"].detach().cpu().numpy().tolist()  # 0-based label into train set

        for im_i, (_input, output) in enumerate(zip(inputs, outputs)):
            json_results = []
            # start_process_time = time.perf_counter()
            # collect predictions for this image
            for out_i, batch_im_i in enumerate(batch_im_ids):
                if im_i != int(batch_im_i):
                    continue

                scene_im_id_split = _input["scene_im_id"].split("/")
                # scene_id = int(scene_im_id_split[0])
                scene_id = scene_im_id_split[0]
                im_id = int(scene_im_id_split[1])

                if "instances" in _input and _input["instances"].has("obj_scores"):
                    inst_id = int(batch_inst_ids[out_i])
                    score = _input["instances"].obj_scores[inst_id]
                else:
                    score = 1.0

                cur_label = batch_labels[out_i]
                if self.train_objs is not None:
                    cls_name = self.train_objs[cur_label]
                else:
                    cls_name = self.obj_names[cur_label]
                obj_id = self.data_ref.obj2id[cls_name]

                # get pose
                cur_jsons = {}
                for refine_i in range(self.n_iter_test + 1):
                    pose_est = pose_est_dict[f"iter{refine_i}"][out_i]
                    cur_jsons[f"iter{refine_i}"] = self.pose_prediction_to_json(
                        pose_est,
                        scene_id,
                        im_id,
                        obj_id=obj_id,
                        score=score,
                        pose_time=output["time"],
                    )

                # estimated poses of all iters for this object
                json_results.append(cur_jsons)

            # output["time"] += time.perf_counter() - start_process_time
            # # process time for this image
            # for item in json_results:
            #     item["time"] = output["time"]
            # extend predictions for this image
            self._predictions.extend(json_results)

    def evaluate(self):
        # bop toolkit eval in subprocess, no return value
        if self._distributed:
            synchronize()
            self._predictions = all_gather(self._predictions)
            self._predictions = list(itertools.chain(*self._predictions))
            if not is_main_process():
                return

        return self._eval_predictions()
        # return copy.deepcopy(self._eval_predictions())

    def _eval_predictions(self):
        """Evaluate self._predictions on 6d pose.

        Return results with the metrics of the tasks.
        """
        self._logger.info("Eval results with BOP toolkit ...")
        # results_all = {"iter0": self._predictions}
        # reformat the predictions of all iters
        results_all = {f"iter{_i}": [] for _i in range(self.n_iter_test + 1)}
        for refine_i in range(self.n_iter_test + 1):
            for pred in self._predictions:
                results_all[f"iter{refine_i}"].append(pred[f"iter{refine_i}"])
        save_and_eval_results(self.cfg, results_all, self._output_dir, obj_ids=self.obj_ids)
        return {}

    def pose_prediction_to_json(self, pose_est, scene_id, im_id, obj_id, score=None, pose_time=-1):
        """
        Args:
            pose_est:
            scene_id (str): the scene id
            img_id (str): the image id
            label: used to get obj_id
            score: confidence
            pose_time:

        Returns:
            dict: the results in BOP evaluation format
        """
        # cfg = self.cfg

        if score is None:  # TODO: add score key in test bbox json file
            score = 1.0
        rot = pose_est[:3, :3]
        trans = pose_est[:3, 3]
        # for standard bop datasets, scene_id and im_id can be obtained from file_name
        result = {
            "scene_id": scene_id,  # if not available, assume 0
            "im_id": im_id,
            "obj_id": obj_id,  # the obj_id in bop datasets
            "score": score,
            "R": to_list(rot),
            "t": to_list(1000 * trans),  # mm
            "time": pose_time,
        }
        return result


def deepim_inference_on_dataset(cfg, model, data_loader, evaluator, amp_test=False):
    """Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.forward` accurately. The model
    will be used in eval mode.

    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use `None` if you only want
            to benchmark, but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    evaluator.reset()

    n_iter_test = cfg.MODEL.DEEPIM.N_ITER_TEST

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    total_process_time = 0
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
                total_process_time = 0

            start_compute_time = time.perf_counter()
            #############################
            # process input ----------------------------------------
            batch = batch_data(cfg, inputs, phase="test")
            if evaluator.train_objs is not None:  # convert label in test data to label in train data
                obj_labels_in_test = batch["obj_cls"].cpu().numpy().tolist()
                obj_labels_in_train = []
                obj_keep_ids = []
                for label_i, test_label in enumerate(obj_labels_in_test):
                    train_label, train_obj_name = evaluator._maybe_adapt_label_cls_name(test_label)
                    if train_label is not None:
                        obj_labels_in_train.append(train_label)
                        obj_keep_ids.append(label_i)
                if len(obj_keep_ids) == 0:  # no label in train found
                    continue
                batch["obj_cls"] = torch.tensor(obj_labels_in_train, device="cuda", dtype=torch.long)
                keep_ids = torch.tensor(obj_keep_ids, device="cuda", dtype=torch.long)
                for _k in batch:
                    if len(batch[_k]) == len(obj_labels_in_test):
                        if isinstance(batch[_k], torch.Tensor):
                            batch[_k] = batch[_k][keep_ids]
                        elif isinstance(batch[_k], list):
                            batch[_k] = [batch[_k][_keep_i] for _keep_i in obj_keep_ids]

            # the input initial pose
            out_dict = {"pose_0": batch["obj_pose_est"]}
            poses_est = None
            for refine_i in range(1, n_iter_test + 1):
                batch_updater(cfg, batch, renderer=evaluator.renderer, poses_est=poses_est, phase="test")
                with autocast(enabled=amp_test):
                    out_dict_i = model(
                        batch["zoom_x"] if "zoom_x" in batch else batch["zoom_x_obs"],
                        x_ren=batch.get("zoom_x_ren", None),
                        init_pose=batch["obj_pose_est"],
                        K_zoom=batch["zoom_K"],
                        obj_class=batch["obj_cls"],
                        # obj_extent=batch.get("obj_extent", None),
                        # roi_coord_2d=batch.get("roi_coord_2d", None),
                        do_loss=False,
                        cur_iter=refine_i,
                    )
                poses_est = out_dict_i[f"pose_{refine_i}"]
                out_dict.update(out_dict_i)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            cur_compute_time = time.perf_counter() - start_compute_time
            total_compute_time += cur_compute_time
            # NOTE: added
            # TODO: add detection time here
            # TODO: time at each iter
            outputs = [{} for _ in range(len(inputs))]  # image-based time
            for _i in range(len(outputs)):
                outputs[_i]["time"] = cur_compute_time

            start_process_time = time.perf_counter()
            evaluator.process(inputs, batch, outputs, out_dict)
            cur_process_time = time.perf_counter() - start_process_time
            total_process_time += cur_process_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    f"Inference done {idx+1}/{total}. {seconds_per_img:.4f} s / img. ETA={str(eta)}",
                    n=5,
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        f"Total inference time: {total_time_str} "
        f"({total_time / (total - num_warmup):.6f} s / img per device, on {num_devices} devices)"
    )
    # pure forward time
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str,
            total_compute_time / (total - num_warmup),
            num_devices,
        )
    )
    # post_process time
    total_process_time_str = str(datetime.timedelta(seconds=int(total_process_time)))
    logger.info(
        "Total inference post process time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_process_time_str,
            total_process_time / (total - num_warmup),
            num_devices,
        )
    )

    results = evaluator.evaluate()  # results is always None
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


def deepim_save_result_of_dataset(
    cfg, model, renderer, data_loader, output_dir, dataset_name, train_objs=None, amp_test=False
):
    """
    Run model (in eval mode) on the data_loader and save predictions
    Args:
        cfg: config
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader)))

    net_cfg = cfg.MODEL.DEEPIM
    n_iter_test = net_cfg.N_ITER_TEST

    # NOTE: dataset name should be the same as TRAIN to get the correct meta
    _metadata = MetadataCatalog.get(dataset_name)
    data_ref = ref.__dict__[_metadata.ref_key]
    obj_names = _metadata.objs
    obj_ids = [data_ref.obj2id[obj_name] for obj_name in obj_names]

    result_name = "results.pkl"
    mmcv.mkdir_or_exist(output_dir)  # NOTE: should be the same as the evaluation output dir
    result_path = osp.join(output_dir, result_name)
    if osp.exists(result_path):
        logger.warning("{} exists, overriding!".format(result_path))

    total = len(data_loader)  # inference data loader must have a fixed length
    result_dict = {}
    VIS = cfg.TEST.VIS  # NOTE: change this for debug/vis
    if VIS:
        import cv2
        from lib.vis_utils.image import vis_image_mask_bbox_cv2, vis_image_bboxes_cv2, grid_show
        from core.utils.my_visualizer import MyVisualizer, _GREY, _GREEN, _BLUE, _RED
        from core.utils.data_utils import crop_resize_by_warp_affine

        # key is [str(obj_id)]["bbox3d_and_center"]
        kpts3d_dict = data_ref.get_keypoints_3d()
        dset_dicts = DatasetCatalog.get(dataset_name)
        scene_im_id_to_gt_index = {d["scene_im_id"]: i for i, d in enumerate(dset_dicts)}

    logging_interval = 50
    num_warmup = min(5, logging_interval - 1, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()
            # process input ----------------------------------------------------------
            batch = batch_data(cfg, inputs, phase="test")
            if train_objs is not None:  # convert label in test data to label in train data
                obj_labels_in_test = batch["obj_cls"].cpu().numpy().tolist()
                obj_labels_in_train = []
                obj_keep_ids = []
                for label_i, test_label in enumerate(obj_labels_in_test):
                    test_obj_name = obj_names[test_label]
                    if test_obj_name not in train_objs:
                        continue

                    train_label = train_objs.index(test_obj_name)
                    obj_labels_in_train.append(train_label)
                    obj_keep_ids.append(label_i)
                if len(obj_keep_ids) == 0:  # no label in train found
                    continue
                batch["obj_cls"] = torch.tensor(obj_labels_in_train, device="cuda", dtype=torch.long)
                keep_ids = torch.tensor(obj_keep_ids, device="cuda", dtype=torch.long)
                for _k in batch:
                    if len(batch[_k]) == len(obj_labels_in_test):
                        if isinstance(batch[_k], torch.Tensor):
                            batch[_k] = batch[_k][keep_ids]
                        elif isinstance(batch[_k], list):
                            batch[_k] = [batch[_k][_keep_i] for _keep_i in obj_keep_ids]

            # NOTE: do model inference -----------------------------------------------
            # the input initial pose
            out_dict = {"pose_0": batch["obj_pose_est"]}
            poses_est = None
            for refine_i in range(1, n_iter_test + 1):
                batch_updater(cfg, batch, renderer=renderer, poses_est=poses_est, phase="test")
                with autocast(enabled=amp_test):
                    out_dict_i = model(
                        batch["zoom_x"] if "zoom_x" in batch else batch["zoom_x_obs"],
                        x_ren=batch.get("zoom_x_ren", None),
                        init_pose=batch["obj_pose_est"],
                        K_zoom=batch["zoom_K"],
                        obj_class=batch["obj_cls"],
                        # obj_extent=batch.get("obj_extent", None),
                        # roi_coord_2d=batch.get("roi_coord_2d", None),
                        do_loss=False,
                        cur_iter=refine_i,
                    )
                poses_est = out_dict_i[f"pose_{refine_i}"]
                out_dict.update(out_dict_i)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            cur_compute_time = time.perf_counter() - start_compute_time
            total_compute_time += cur_compute_time

            # NOTE: process results ----------------------------------------
            pose_est_dict = {}
            for _i in range(n_iter_test + 1):
                pose_est_dict[f"iter{_i}"] = out_dict[f"pose_{_i}"].detach().cpu().numpy()
            batch_im_ids = batch["im_id"].detach().cpu().numpy().tolist()
            batch_inst_ids = batch["inst_id"].detach().cpu().numpy().tolist()
            # 0-based label into train set (already adapted)
            batch_labels = batch["obj_cls"].detach().cpu().numpy().tolist()
            if VIS:
                batch_Ks = batch["K"].detach().cpu().numpy()

            for im_i, _input in enumerate(inputs):
                if VIS:
                    image_path = _input["file_name"]
                    image = mmcv.imread(image_path, "color")
                # collect predictions for this image
                for out_i, batch_im_i in enumerate(batch_im_ids):
                    if im_i != int(batch_im_i):
                        continue

                    scene_im_id = _input["scene_im_id"]
                    scene_im_id_split = scene_im_id.split("/")
                    scene_id = scene_im_id_split[0]
                    im_id = int(scene_im_id_split[1])

                    if "instances" in _input and _input["instances"].has("obj_scores"):
                        inst_id = int(batch_inst_ids[out_i])
                        score = float(_input["instances"].obj_scores[inst_id])
                    else:
                        score = 1.0

                    cur_label = batch_labels[out_i]
                    if train_objs is not None:
                        cls_name = train_objs[cur_label]
                    else:
                        cls_name = obj_names[cur_label]
                    obj_id = data_ref.obj2id[cls_name]

                    # collect cur res
                    cur_res = {
                        "obj_id": obj_id,
                        "score": score,
                    }
                    # NOTE: add bbox_est (bbox_det, xyxy) from detections if available
                    if "instances" in _input and _input["instances"].has("obj_boxes_det"):
                        log_first_n(logging.INFO, "obj_boxes_det is available", n=1)
                        inst_id = int(batch_inst_ids[out_i])
                        bbox_det_xyxy = (
                            _input["instances"]
                            .obj_boxes_det[inst_id]
                            .tensor.cpu()
                            .numpy()
                            .reshape(
                                4,
                            )
                        )
                        cur_res["bbox_det_xyxy"] = bbox_det_xyxy

                    for refine_i in range(n_iter_test + 1):
                        pose_est_i = pose_est_dict[f"iter{refine_i}"][out_i]
                        cur_res[f"pose_{refine_i}"] = pose_est_i

                    if scene_im_id not in result_dict:
                        result_dict[scene_im_id] = []
                    result_dict[scene_im_id].append(cur_res)

                    if VIS:  # vis -----------------------------------------------------------
                        vis_dict = {}
                        if "bbox_det_xyxy" in cur_res:
                            img_bbox_vis = vis_image_bboxes_cv2(
                                image, [bbox_det_xyxy], box_color=(200, 200, 10)
                            )  # cyan
                            vis_dict["img_bbox_det"] = img_bbox_vis[:, :, ::-1]

                        K = batch_Ks[out_i]
                        kpt3d = kpts3d_dict[str(obj_id)]["bbox3d_and_center"]
                        pose_est_0 = cur_res[f"pose_{0}"]
                        kpt2d_0 = misc.project_pts(kpt3d, K, pose_est_0[:3, :3], pose_est_0[:3, 3])

                        # gt pose
                        gt_idx = scene_im_id_to_gt_index[scene_im_id]
                        gt_dict = dset_dicts[gt_idx]
                        gt_annos = gt_dict["annotations"]
                        # find the gt anno ---------------
                        found_gt = False
                        for gt_anno in gt_annos:
                            gt_label = gt_anno["category_id"]
                            gt_obj = obj_names[gt_label]
                            gt_obj_id = data_ref.obj2id[gt_obj]
                            if obj_id == gt_obj_id:
                                found_gt = True
                                gt_pose = gt_anno["pose"]
                                break
                        if not found_gt:
                            kpt2d_gt = None
                        else:
                            kpt2d_gt = misc.project_pts(kpt3d, K, gt_pose[:3, :3], gt_pose[:3, 3])

                        maxx, maxy, minx, miny = 0, 0, 1000, 1000
                        for i in range(len(kpt2d_0)):
                            maxx, maxy, minx, miny = (
                                max(maxx, kpt2d_0[i][0]),
                                max(maxy, kpt2d_0[i][1]),
                                min(minx, kpt2d_0[i][0]),
                                min(miny, kpt2d_0[i][1]),
                            )
                            if kpt2d_gt is not None:
                                maxx, maxy, minx, miny = (
                                    max(maxx, kpt2d_gt[i][0]),
                                    max(maxy, kpt2d_gt[i][1]),
                                    min(minx, kpt2d_gt[i][0]),
                                    min(miny, kpt2d_gt[i][1]),
                                )
                        center_0 = np.array([(minx + maxx) / 2, (miny + maxy) / 2])
                        scale_0 = max(maxx - minx, maxy - miny) * 1.5  # * 3  # + 10
                        CROP_SIZE = 256
                        im_zoom = crop_resize_by_warp_affine(image, center_0, scale_0, CROP_SIZE)

                        zoom_kpt2d_0 = kpt2d_0.copy()
                        for i in range(len(kpt2d_0)):
                            zoom_kpt2d_0[i][0] = (kpt2d_0[i][0] - (center_0[0] - scale_0 / 2)) * CROP_SIZE / scale_0
                            zoom_kpt2d_0[i][1] = (kpt2d_0[i][1] - (center_0[1] - scale_0 / 2)) * CROP_SIZE / scale_0

                        if kpt2d_gt is not None:
                            zoom_kpt2d_gt = kpt2d_gt.copy()
                            for i in range(len(kpt2d_gt)):
                                zoom_kpt2d_gt[i][0] = (
                                    (kpt2d_gt[i][0] - (center_0[0] - scale_0 / 2)) * CROP_SIZE / scale_0
                                )
                                zoom_kpt2d_gt[i][1] = (
                                    (kpt2d_gt[i][1] - (center_0[1] - scale_0 / 2)) * CROP_SIZE / scale_0
                                )

                        # TODO: maybe vis contour
                        for refine_i in range(1, n_iter_test + 1):
                            pose_est_i = cur_res[f"pose_{refine_i}"]
                            kpt2d_i = misc.project_pts(kpt3d, K, pose_est_i[:3, :3], pose_est_i[:3, 3])

                            zoom_kpt2d_i = kpt2d_i.copy()
                            for i in range(len(kpt2d_i)):
                                zoom_kpt2d_i[i][0] = (kpt2d_i[i][0] - (center_0[0] - scale_0 / 2)) * CROP_SIZE / scale_0
                                zoom_kpt2d_i[i][1] = (kpt2d_i[i][1] - (center_0[1] - scale_0 / 2)) * CROP_SIZE / scale_0

                            visualizer = MyVisualizer(im_zoom[:, :, ::-1], _metadata)
                            linewidth = 3
                            if kpt2d_gt is not None:
                                visualizer.draw_bbox3d_and_center(
                                    zoom_kpt2d_gt,
                                    top_color=_BLUE,
                                    bottom_color=_GREY,
                                    linewidth=linewidth,
                                    draw_center=True,
                                )
                            visualizer.draw_bbox3d_and_center(
                                zoom_kpt2d_0, top_color=_RED, bottom_color=_GREY, linewidth=linewidth, draw_center=True
                            )
                            visualizer.draw_bbox3d_and_center(
                                zoom_kpt2d_i,
                                top_color=_GREEN,
                                bottom_color=_GREY,
                                linewidth=linewidth,
                                draw_center=True,
                            )
                            im_init_refine_i = visualizer.get_output().get_image()
                            vis_dict["zoom_im_init_refine_{}".format(refine_i)] = im_init_refine_i
                        show_titles = [_k for _k, _v in vis_dict.items()]
                        show_ims = [_v for _k, _v in vis_dict.items()]
                        ncol = 3
                        nrow = int(np.ceil(len(show_ims) / ncol))
                        grid_show(show_ims, show_titles, row=nrow, col=ncol)

                    # end vis -------------------------------------------------------------------------
            # -----------------------------------------------------------------------------------------
            if (idx + 1) % logging_interval == 0:
                duration = time.perf_counter() - start_time
                seconds_per_img = duration / (idx + 1 - num_warmup)
                eta = datetime.timedelta(seconds=int(seconds_per_img * (total - num_warmup) - duration))
                logger.info(
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(idx + 1, total, seconds_per_img, str(eta))
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = int(time.perf_counter() - start_time)
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    mmcv.dump(result_dict, result_path)
    logger.info("Results saved to {}".format(result_path))
