# -*- coding: utf-8 -*-
"""inference on dataset; save results; evaluate with custom evaluation
funcs."""
import itertools
import logging
import os.path as osp
import random
import time
from collections import OrderedDict

import cv2
import mmcv
import numpy as np
import torch
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from tabulate import tabulate
from tqdm import tqdm
from transforms3d.quaternions import quat2mat

cur_dir = osp.dirname(osp.abspath(__file__))
import ref
from core.utils.my_comm import all_gather, is_main_process, synchronize
from core.utils.pose_utils import get_closest_rot
from core.utils.my_visualizer import MyVisualizer, _GREEN, _GREY
from core.utils.data_utils import crop_resize_by_warp_affine
from lib.pysixd import inout, misc
from lib.pysixd.pose_error import add, adi, arp_2d, re, te
from lib.vis_utils.image import grid_show, vis_image_bboxes_cv2

from core.deepim.engine.engine_utils import get_out_coor, get_out_mask

PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../../.."))


class DeepIM_EvaluatorCustom(DatasetEvaluator):
    """custom evaluation of 6d pose.

    Assume single instance!!!
    """

    def __init__(
        self,
        cfg,
        dataset_name,
        distributed,
        output_dir,
        train_objs=None,
        renderer=None,
    ):
        self.cfg = cfg
        self.n_iter_test = cfg.MODEL.DEEPIM.N_ITER_TEST

        self._distributed = distributed
        self._output_dir = output_dir
        mmcv.mkdir_or_exist(output_dir)

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        # if test objs are just a subset of train objs
        self.train_objs = train_objs
        self.renderer = renderer

        self.dataset_name = dataset_name
        self._metadata = MetadataCatalog.get(dataset_name)
        self.data_ref = ref.__dict__[self._metadata.ref_key]
        self.obj_names = self._metadata.objs
        self.obj_ids = [self.data_ref.obj2id[obj_name] for obj_name in self.obj_names]
        # with contextlib.redirect_stdout(io.StringIO()):
        #     self._coco_api = COCO(self._metadata.json_file)
        self.model_paths = [
            osp.join(self.data_ref.model_eval_dir, "obj_{:06d}.ply".format(obj_id)) for obj_id in self.obj_ids
        ]
        self.diameters = [self.data_ref.diameters[self.data_ref.objects.index(obj_name)] for obj_name in self.obj_names]
        self.models_3d = [
            inout.load_ply(model_path, vertex_scale=self.data_ref.vertex_scale) for model_path in self.model_paths
        ]

        self.eval_precision = cfg.VAL.get("EVAL_PRECISION", False)
        self._logger.info(f"eval precision: {self.eval_precision}")
        # eval cached
        self.use_cache = False
        if cfg.VAL.EVAL_CACHED or cfg.VAL.EVAL_PRINT_ONLY:
            self.use_cache = True
            for refine_i in range(self.n_iter_test + 1):
                if self.eval_precision:
                    self._eval_predictions_precision(refine_i)
                else:
                    self._eval_predictions(refine_i)  # recall
            exit(0)

    def reset(self):
        self._predictions = []
        # when evaluate, transform the list to dict
        self._predictions_dict = OrderedDict()

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
            outputs:
            out_dict: the predictions of the model
        """
        pose_est_dict = {}
        for _i in range(self.n_iter_test + 1):
            pose_est_dict[f"iter{_i}"] = out_dict[f"pose_{_i}"].detach().cpu().numpy()
        batch_im_ids = batch["im_id"].detach().cpu().numpy().tolist()
        batch_inst_ids = batch["inst_id"].detach().cpu().numpy().tolist()
        batch_labels = batch["obj_cls"].detach().cpu().numpy().tolist()  # 0-based label into train set

        for im_i, (_input, output) in enumerate(zip(inputs, outputs)):
            # start_process_time = time.perf_counter()
            # collect predictions for this image
            for out_i, batch_im_i in enumerate(batch_im_ids):
                if im_i != int(batch_im_i):
                    continue
                file_name = _input["file_name"]
                scene_im_id_split = _input["scene_im_id"].split("/")
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
                # obj_id = self.data_ref.obj2id[cls_name]

                # get pose
                cur_result = {}
                for refine_i in range(self.n_iter_test + 1):
                    pose_est = pose_est_dict[f"iter{refine_i}"][out_i]
                    cur_result[f"iter{refine_i}"] = {
                        "score": score,
                        "R": pose_est[:3, :3],
                        "t": pose_est[:3, 3],
                        "time": output["time"],
                    }
                # output["time"] += time.perf_counter() - start_process_time

                self._predictions.append((cls_name, file_name, cur_result))

    def _preds_list_to_dict(self):
        # list of tuple to dict
        for cls_name, file_name, result in self._predictions:
            if cls_name not in self._predictions_dict:
                self._predictions_dict[cls_name] = OrderedDict()
            self._predictions_dict[cls_name][file_name] = result

    def evaluate(self):
        if self._distributed:
            synchronize()
            self._predictions = all_gather(self._predictions)
            self._predictions = list(itertools.chain(*self._predictions))

            if not is_main_process():
                return

        self._preds_list_to_dict()
        eval_res = {}
        for refine_i in range(self.n_iter_test + 1):
            if self.eval_precision:
                eval_res.update(self._eval_predictions_precision(refine_i))  # precision
            eval_res.update(self._eval_predictions(refine_i))  # recall
        return eval_res
        # return copy.deepcopy(self._eval_predictions())

    def get_gts(self):
        # NOTE: it is cached by dataset dicts loader
        self.gts = OrderedDict()

        dataset_dicts = DatasetCatalog.get(self.dataset_name)
        self._logger.info("load gts of {}".format(self.dataset_name))
        for im_dict in tqdm(dataset_dicts):
            file_name = im_dict["file_name"]
            annos = im_dict["annotations"]
            K = im_dict["cam"]
            for anno in annos:
                quat = anno["quat"]
                R = quat2mat(quat)
                trans = anno["trans"]
                obj_name = self._metadata.objs[anno["category_id"]]
                if obj_name not in self.gts:
                    self.gts[obj_name] = OrderedDict()
                self.gts[obj_name][file_name] = {"R": R, "t": trans, "K": K}

    def _eval_predictions(self, cur_iter=0):
        """Evaluate self._predictions on 6d pose.

        Return results with the metrics of the tasks.
        """
        self._logger.info(f"Eval recalls of results at iter={cur_iter}...")
        cfg = self.cfg
        method_name = f"{cfg.EXP_ID.replace('_', '-')}"
        cache_path = osp.join(self._output_dir, f"{method_name}_{self.dataset_name}_preds.pkl")
        if cur_iter == 0:  # only load or dump results at iter0
            if osp.exists(cache_path) and self.use_cache:
                self._logger.info("load cached predictions")
                self._predictions_dict = mmcv.load(cache_path)
            else:
                if hasattr(self, "_predictions_dict"):
                    mmcv.dump(self._predictions_dict, cache_path)
                else:
                    raise RuntimeError("Please run inference first")
            self.get_gts()

        recalls = OrderedDict()
        errors = OrderedDict()

        error_names = ["ad", "re", "te", "proj"]
        # yapf: disable
        metric_names = [
            "ad_2", "ad_5", "ad_10",
            "rete_2", "rete_5", "rete_10",
            "re_2", "re_5", "re_10",
            "te_2", "te_5", "te_10",
            "proj_2", "proj_5", "proj_10",
        ]
        # yapf: enable

        for obj_name in self.gts:
            if obj_name not in self._predictions_dict:
                continue
            cur_label = self.obj_names.index(obj_name)
            if obj_name not in recalls:
                recalls[obj_name] = OrderedDict()
                for metric_name in metric_names:
                    recalls[obj_name][metric_name] = []

            if obj_name not in errors:
                errors[obj_name] = OrderedDict()
                for err_name in error_names:
                    errors[obj_name][err_name] = []

            #################
            obj_gts = self.gts[obj_name]
            obj_preds = self._predictions_dict[obj_name]
            for file_name, gt_anno in obj_gts.items():
                if file_name not in obj_preds:  # no pred found
                    for metric_name in metric_names:
                        recalls[obj_name][metric_name].append(0.0)
                    continue
                # compute each metric
                R_pred = obj_preds[file_name][f"iter{cur_iter}"]["R"]
                t_pred = obj_preds[file_name][f"iter{cur_iter}"]["t"]

                R_gt = gt_anno["R"]
                t_gt = gt_anno["t"]

                t_error = te(t_pred, t_gt)

                if obj_name in cfg.DATASETS.SYM_OBJS:
                    R_gt_sym = get_closest_rot(R_pred, R_gt, self._metadata.sym_infos[cur_label])
                    r_error = re(R_pred, R_gt_sym)

                    proj_2d_error = arp_2d(
                        R_pred,
                        t_pred,
                        R_gt_sym,
                        t_gt,
                        pts=self.models_3d[cur_label]["pts"],
                        K=gt_anno["K"],
                    )

                    ad_error = adi(
                        R_pred,
                        t_pred,
                        R_gt,
                        t_gt,
                        pts=self.models_3d[self.obj_names.index(obj_name)]["pts"],
                    )
                else:
                    r_error = re(R_pred, R_gt)

                    proj_2d_error = arp_2d(
                        R_pred,
                        t_pred,
                        R_gt,
                        t_gt,
                        pts=self.models_3d[cur_label]["pts"],
                        K=gt_anno["K"],
                    )

                    ad_error = add(
                        R_pred,
                        t_pred,
                        R_gt,
                        t_gt,
                        pts=self.models_3d[self.obj_names.index(obj_name)]["pts"],
                    )

                #########
                errors[obj_name]["ad"].append(ad_error)
                errors[obj_name]["re"].append(r_error)
                errors[obj_name]["te"].append(t_error)
                errors[obj_name]["proj"].append(proj_2d_error)
                ############
                recalls[obj_name]["ad_2"].append(float(ad_error < 0.02 * self.diameters[cur_label]))
                recalls[obj_name]["ad_5"].append(float(ad_error < 0.05 * self.diameters[cur_label]))
                recalls[obj_name]["ad_10"].append(float(ad_error < 0.1 * self.diameters[cur_label]))
                # deg, cm
                recalls[obj_name]["rete_2"].append(float(r_error < 2 and t_error < 0.02))
                recalls[obj_name]["rete_5"].append(float(r_error < 5 and t_error < 0.05))
                recalls[obj_name]["rete_10"].append(float(r_error < 10 and t_error < 0.1))

                recalls[obj_name]["re_2"].append(float(r_error < 2))
                recalls[obj_name]["re_5"].append(float(r_error < 5))
                recalls[obj_name]["re_10"].append(float(r_error < 10))

                recalls[obj_name]["te_2"].append(float(t_error < 0.02))
                recalls[obj_name]["te_5"].append(float(t_error < 0.05))
                recalls[obj_name]["te_10"].append(float(t_error < 0.1))
                # px
                recalls[obj_name]["proj_2"].append(float(proj_2d_error < 2))
                recalls[obj_name]["proj_5"].append(float(proj_2d_error < 5))
                recalls[obj_name]["proj_10"].append(float(proj_2d_error < 10))

        # summarize
        obj_names = sorted(list(recalls.keys()))
        header = ["objects"] + obj_names + [f"Avg({len(obj_names)})"]
        big_tab = [header]
        for metric_name in metric_names:
            line = [metric_name]
            this_line_res = []
            for obj_name in obj_names:
                res = recalls[obj_name][metric_name]
                if len(res) > 0:
                    line.append(f"{100 * np.mean(res):.2f}")
                    this_line_res.append(np.mean(res))
                else:
                    line.append(0.0)
                    this_line_res.append(0.0)
            # average
            if len(obj_names) > 0:
                line.append(f"{100 * np.mean(this_line_res):.2f}")
            big_tab.append(line)

        for error_name in ["re", "te"]:
            line = [error_name]
            this_line_res = []
            for obj_name in obj_names:
                res = errors[obj_name][error_name]
                if len(res) > 0:
                    line.append(f"{np.mean(res):.2f}")
                    this_line_res.append(np.mean(res))
                else:
                    line.append(float("nan"))
                    this_line_res.append(float("nan"))
            # mean
            if len(obj_names) > 0:
                line.append(f"{np.mean(this_line_res):.2f}")
            big_tab.append(line)
        ### log big tag
        self._logger.info(f"recalls at iter{cur_iter}")
        res_log_tab_str = tabulate(
            big_tab,
            tablefmt="plain",
            # floatfmt=floatfmt
        )
        self._logger.info("\n{}".format(res_log_tab_str))
        errors_cache_path = osp.join(
            self._output_dir,
            f"{method_name}_{self.dataset_name}_errors_iter{cur_iter}.pkl",
        )
        recalls_cache_path = osp.join(
            self._output_dir,
            f"{method_name}_{self.dataset_name}_recalls_iter{cur_iter}.pkl",
        )
        mmcv.dump(errors, errors_cache_path)
        mmcv.dump(recalls, recalls_cache_path)

        dump_tab_name = osp.join(
            self._output_dir,
            f"{method_name}_{self.dataset_name}_tab_iter{cur_iter}.txt",
        )
        with open(dump_tab_name, "w") as f:
            f.write("{}\n".format(res_log_tab_str))

        if self._distributed:
            self._logger.warning("\n The current evaluation on multi-gpu is not correct, run with single-gpu instead.")

        return {}

    def _eval_predictions_precision(self, cur_iter=0):
        """NOTE: eval precision instead of recall
        Evaluate self._predictions on 6d pose.
        Return results with the metrics of the tasks.
        """
        self._logger.info(f"Eval precisions of results at iter={cur_iter}...")
        cfg = self.cfg
        if cur_iter == 0:
            method_name = f"{cfg.EXP_ID.replace('_', '-')}"
            cache_path = osp.join(
                self._output_dir,
                f"{method_name}_{self.dataset_name}_preds.pkl",
            )
            if osp.exists(cache_path) and self.use_cache:
                self._logger.info("load cached predictions")
                self._predictions_dict = mmcv.load(cache_path)
            else:
                if hasattr(self, "_predictions_dict"):
                    mmcv.dump(self._predictions_dict, cache_path)
                else:
                    raise RuntimeError("Please run inference first")
            self.get_gts()

        precisions = OrderedDict()
        errors = OrderedDict()

        error_names = ["ad", "re", "te", "proj"]
        # yapf: disable
        metric_names = [
            "ad_2", "ad_5", "ad_10",
            "rete_2", "rete_5", "rete_10",
            "re_2", "re_5", "re_10",
            "te_2", "te_5", "te_10",
            "proj_2", "proj_5", "proj_10",
        ]
        # yapf: enable

        for obj_name in self.gts:
            if obj_name not in self._predictions_dict:
                continue
            cur_label = self.obj_names.index(obj_name)
            if obj_name not in precisions:
                precisions[obj_name] = OrderedDict()
                for metric_name in metric_names:
                    precisions[obj_name][metric_name] = []

            if obj_name not in errors:
                errors[obj_name] = OrderedDict()
                for err_name in error_names:
                    errors[obj_name][err_name] = []

            #################
            obj_gts = self.gts[obj_name]
            obj_preds = self._predictions_dict[obj_name]
            for file_name, gt_anno in obj_gts.items():
                # compute precision as in DPOD paper
                if file_name not in obj_preds:  # no pred found
                    # NOTE: just ignore undetected
                    continue
                # compute each metric
                R_pred = obj_preds[file_name][f"iter{cur_iter}"]["R"]
                t_pred = obj_preds[file_name][f"iter{cur_iter}"]["t"]

                R_gt = gt_anno["R"]
                t_gt = gt_anno["t"]

                t_error = te(t_pred, t_gt)

                if obj_name in cfg.DATASETS.SYM_OBJS:
                    R_gt_sym = get_closest_rot(R_pred, R_gt, self._metadata.sym_infos[cur_label])
                    r_error = re(R_pred, R_gt_sym)

                    proj_2d_error = arp_2d(
                        R_pred,
                        t_pred,
                        R_gt_sym,
                        t_gt,
                        pts=self.models_3d[cur_label]["pts"],
                        K=gt_anno["K"],
                    )

                    ad_error = adi(
                        R_pred,
                        t_pred,
                        R_gt,
                        t_gt,
                        pts=self.models_3d[self.obj_names.index(obj_name)]["pts"],
                    )
                else:
                    r_error = re(R_pred, R_gt)

                    proj_2d_error = arp_2d(
                        R_pred,
                        t_pred,
                        R_gt,
                        t_gt,
                        pts=self.models_3d[cur_label]["pts"],
                        K=gt_anno["K"],
                    )

                    ad_error = add(
                        R_pred,
                        t_pred,
                        R_gt,
                        t_gt,
                        pts=self.models_3d[self.obj_names.index(obj_name)]["pts"],
                    )

                #########
                errors[obj_name]["ad"].append(ad_error)
                errors[obj_name]["re"].append(r_error)
                errors[obj_name]["te"].append(t_error)
                errors[obj_name]["proj"].append(proj_2d_error)
                ############
                precisions[obj_name]["ad_2"].append(float(ad_error < 0.02 * self.diameters[cur_label]))
                precisions[obj_name]["ad_5"].append(float(ad_error < 0.05 * self.diameters[cur_label]))
                precisions[obj_name]["ad_10"].append(float(ad_error < 0.1 * self.diameters[cur_label]))
                # deg, cm
                precisions[obj_name]["rete_2"].append(float(r_error < 2 and t_error < 0.02))
                precisions[obj_name]["rete_5"].append(float(r_error < 5 and t_error < 0.05))
                precisions[obj_name]["rete_10"].append(float(r_error < 10 and t_error < 0.1))

                precisions[obj_name]["re_2"].append(float(r_error < 2))
                precisions[obj_name]["re_5"].append(float(r_error < 5))
                precisions[obj_name]["re_10"].append(float(r_error < 10))

                precisions[obj_name]["te_2"].append(float(t_error < 0.02))
                precisions[obj_name]["te_5"].append(float(t_error < 0.05))
                precisions[obj_name]["te_10"].append(float(t_error < 0.1))
                # px
                precisions[obj_name]["proj_2"].append(float(proj_2d_error < 2))
                precisions[obj_name]["proj_5"].append(float(proj_2d_error < 5))
                precisions[obj_name]["proj_10"].append(float(proj_2d_error < 10))

        # summarize
        obj_names = sorted(list(precisions.keys()))
        header = ["objects"] + obj_names + [f"Avg({len(obj_names)})"]
        big_tab = [header]
        for metric_name in metric_names:
            line = [metric_name]
            this_line_res = []
            for obj_name in obj_names:
                res = precisions[obj_name][metric_name]
                if len(res) > 0:
                    line.append(f"{100 * np.mean(res):.2f}")
                    this_line_res.append(np.mean(res))
                else:
                    line.append(0.0)
                    this_line_res.append(0.0)
            # mean
            if len(obj_names) > 0:
                line.append(f"{100 * np.mean(this_line_res):.2f}")
            big_tab.append(line)

        for error_name in ["re", "te"]:
            line = [error_name]
            this_line_res = []
            for obj_name in obj_names:
                res = errors[obj_name][error_name]
                if len(res) > 0:
                    line.append(f"{np.mean(res):.2f}")
                    this_line_res.append(np.mean(res))
                else:
                    line.append(float("nan"))
                    this_line_res.append(float("nan"))
            # mean
            if len(obj_names) > 0:
                line.append(f"{np.mean(this_line_res):.2f}")
            big_tab.append(line)
        ### log big table
        self._logger.info(f"precisions at iter{cur_iter}")
        res_log_tab_str = tabulate(
            big_tab,
            tablefmt="plain",
            # floatfmt=floatfmt
        )
        self._logger.info("\n{}".format(res_log_tab_str))
        errors_cache_path = osp.join(
            self._output_dir,
            f"{method_name}_{self.dataset_name}_errors_iter{cur_iter}.pkl",
        )
        recalls_cache_path = osp.join(
            self._output_dir,
            f"{method_name}_{self.dataset_name}_precisions_iter{cur_iter}.pkl",
        )
        self._logger.info(f"{errors_cache_path}")
        self._logger.info(f"{recalls_cache_path}")
        mmcv.dump(errors, errors_cache_path)
        mmcv.dump(precisions, recalls_cache_path)

        dump_tab_name = osp.join(
            self._output_dir,
            f"{method_name}_{self.dataset_name}_tab_precisions_iter{cur_iter}.txt",
        )
        with open(dump_tab_name, "w") as f:
            f.write("{}\n".format(res_log_tab_str))
        if self._distributed:
            self._logger.warning("\n The current evaluation on multi-gpu is not correct, run with single-gpu instead.")
        return {}
