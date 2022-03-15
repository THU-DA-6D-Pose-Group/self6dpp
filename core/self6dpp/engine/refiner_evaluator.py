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
from core.utils.my_comm import (
    all_gather,
    get_world_size,
    is_main_process,
    synchronize,
)
from detectron2.data import MetadataCatalog
from detectron2.evaluation import (
    DatasetEvaluator,
    DatasetEvaluators,
    inference_context,
)
from detectron2.utils.logger import log_every_n_seconds
from lib.pysixd import inout, misc
from lib.pysixd.pose_error import te
from lib.utils.mask_utils import binary_mask_to_rle
from lib.vis_utils.image import grid_show, vis_image_bboxes_cv2
from torch.cuda.amp import autocast
from transforms3d.quaternions import quat2mat

from .refiner_batching import batch_data, batch_updater
from .test_utils import eval_cached_results, save_and_eval_results, to_list


class Refiner_Evaluator(DatasetEvaluator):
    """use bop toolkit to evaluate."""

    def __init__(
        self,
        cfg,
        dataset_name,
        distributed,
        output_dir,
        train_objs=None,
        renderer=None,
        ren_models=None,
    ):
        self.cfg = cfg
        self.n_iter_test = cfg.MODEL.DEEPIM.N_ITER_TEST

        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        # if test objs are just a subset of train objs
        self.train_objs = train_objs
        self.renderer = renderer
        self.ren_models = ren_models

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


def refiner_inference_on_dataset(cfg, model, data_loader, evaluator, amp_test=False):
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
            # process input
            batch = batch_data(cfg, inputs, phase="test")
            if evaluator.train_objs is not None:  # convert label in test data to label in train data
                obj_labels_in_test = batch["obj_cls"].cpu().numpy().tolist()
                obj_labels_in_train = []
                obj_keep_ids = []
                for label_i, test_label in enumerate(obj_labels_in_test):
                    (
                        train_label,
                        train_obj_name,
                    ) = evaluator._maybe_adapt_label_cls_name(test_label)
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
            for refine_i in range(1, cfg.MODEL.DEEPIM.N_ITER_TEST + 1):
                batch_updater(
                    cfg,
                    batch,
                    renderer=evaluator.renderer,
                    poses_est=poses_est,
                    ren_models=evaluator.ren_models,
                    phase="test",
                )
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
