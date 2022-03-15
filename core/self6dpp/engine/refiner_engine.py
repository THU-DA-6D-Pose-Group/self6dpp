import logging
import os
import os.path as osp
import torch
from torch.cuda.amp import autocast, GradScaler
import mmcv
import time
import numpy as np
from collections import OrderedDict

from detectron2.utils.events import EventStorage
from detectron2.checkpoint import PeriodicCheckpointer
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
)

from detectron2.data.common import AspectRatioGroupedDataset
from detectron2.data import MetadataCatalog

from lib.utils.utils import dprint

from core.utils import solver_utils
import core.utils.my_comm as comm
from core.utils.my_checkpoint import MyCheckpointer
from core.utils.my_writer import (
    MyCommonMetricPrinter,
    MyJSONWriter,
    MyTensorboardXWriter,
)
from core.utils.data_utils import denormalize_image

# change below for different methods
from core.self6dpp.datasets.data_loader_refiner import (
    build_refiner_train_loader,
    build_refiner_test_loader,
)
from .refiner_engine_utils import get_out_mask, vis_batch
from .refiner_batching import batch_data, batch_updater
from .refiner_evaluator import refiner_inference_on_dataset, Refiner_Evaluator
from .refiner_custom_evaluator import Refiner_EvaluatorCustom

logger = logging.getLogger(__name__)


def get_evaluator(cfg, dataset_name, output_folder=None, renderer=None, ren_models=None):
    """Create evaluator(s) for a given dataset.

    This uses the special metadata "evaluator_type" associated with each
    builtin dataset. For your own dataset, you can simply create an
    evaluator manually in your script and do not have to worry about the
    hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = osp.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        assert (
            torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        assert (
            torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)

    _distributed = comm.get_world_size() > 1
    dataset_meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    train_obj_names = dataset_meta.objs
    if evaluator_type == "bop":
        refiner_eval_cls = Refiner_Evaluator if cfg.VAL.get("USE_BOP", False) else Refiner_EvaluatorCustom
        return refiner_eval_cls(
            cfg,
            dataset_name,
            distributed=_distributed,
            output_dir=output_folder,
            train_objs=train_obj_names,
            renderer=renderer,
            ren_models=ren_models,
        )

    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_test(cfg, model, renderer=None, ren_models=None, epoch=None, iteration=None):
    results = OrderedDict()
    model_name = osp.basename(cfg.MODEL.WEIGHTS).split(".")[0]
    for dataset_name in cfg.DATASETS.TEST:
        if epoch is not None and iteration is not None:
            eval_out_dir = osp.join(
                cfg.OUTPUT_DIR,
                f"inference_epoch_{epoch}_iter_{iteration}",
                dataset_name,
            )
        else:
            eval_out_dir = osp.join(cfg.OUTPUT_DIR, f"inference_{model_name}", dataset_name)
        evaluator = get_evaluator(
            cfg,
            dataset_name,
            eval_out_dir,
            renderer=renderer,
            ren_models=ren_models,
        )

        data_loader = build_refiner_test_loader(cfg, dataset_name, train_objs=evaluator.train_objs)
        results_i = refiner_inference_on_dataset(cfg, model, data_loader, evaluator, amp_test=cfg.TEST.AMP_TEST)
        results[dataset_name] = results_i
        # if comm.is_main_process():
        #     logger.info("Evaluation results for {} in csv format:".format(dataset_name))
        #     print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def get_tbx_event_writer(out_dir, backup=False):
    tb_logdir = osp.join(out_dir, "tb")
    mmcv.mkdir_or_exist(tb_logdir)
    if backup and comm.is_main_process():
        old_tb_logdir = osp.join(out_dir, "tb_old")
        mmcv.mkdir_or_exist(old_tb_logdir)
        os.system("mv -v {} {}".format(osp.join(tb_logdir, "events.*"), old_tb_logdir))

    tbx_event_writer = MyTensorboardXWriter(tb_logdir, backend="tensorboardX")
    return tbx_event_writer


def do_train(cfg, args, model, optimizer, renderer, ren_models=None, resume=False):
    net_cfg = cfg.MODEL.DEEPIM
    model.train()

    # some basic settings =========================
    # dataset_meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    # obj_names = dataset_meta.objs

    # load data ===================================
    train_dset_names = cfg.DATASETS.TRAIN
    data_loader = build_refiner_train_loader(cfg, train_dset_names)
    data_loader_iter = iter(data_loader)

    # load 2nd train dataloader if needed
    train_2_dset_names = cfg.DATASETS.get("TRAIN2", ())
    train_2_ratio = cfg.DATASETS.get("TRAIN2_RATIO", 0.0)
    if train_2_ratio > 0.0 and len(train_2_dset_names) > 0:
        data_loader_2 = build_refiner_train_loader(cfg, train_2_dset_names)
        data_loader_2_iter = iter(data_loader_2)
    else:
        data_loader_2 = None
        data_loader_2_iter = None

    images_per_batch = cfg.SOLVER.IMS_PER_BATCH
    if isinstance(data_loader, AspectRatioGroupedDataset):
        dataset_len = len(data_loader.dataset.dataset)
        if data_loader_2 is not None:
            dataset_len += len(data_loader_2.dataset.dataset)
        iters_per_epoch = dataset_len // images_per_batch
    else:
        dataset_len = len(data_loader.dataset)
        if data_loader_2 is not None:
            dataset_len += len(data_loader_2.dataset)
        iters_per_epoch = dataset_len // images_per_batch
    max_iter = cfg.SOLVER.TOTAL_EPOCHS * iters_per_epoch
    dprint("images_per_batch: ", images_per_batch)
    dprint("dataset length: ", dataset_len)
    dprint("iters per epoch: ", iters_per_epoch)
    dprint("total iters: ", max_iter)
    scheduler = solver_utils.build_lr_scheduler(cfg, optimizer, total_iters=max_iter)

    AMP_ON = cfg.SOLVER.AMP.ENABLED
    logger.info(f"AMP enabled: {AMP_ON}")
    grad_scaler = GradScaler()

    # resume or load model ===================================
    checkpointer = MyCheckpointer(
        model,
        cfg.OUTPUT_DIR,
        optimizer=optimizer,
        scheduler=scheduler,
        gradscaler=grad_scaler,
        save_to_disk=comm.is_main_process(),
    )
    start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1

    if comm._USE_HVD:  # hvd may be not available, so do not use the one in args
        import horovod.torch as hvd

        # not needed
        # start_iter = hvd.broadcast(torch.tensor(start_iter), root_rank=0, name="start_iter").item()

        # Horovod: broadcast parameters & optimizer state.
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        # Horovod: (optional) compression algorithm.
        compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
        optimizer = hvd.DistributedOptimizer(
            optimizer,
            named_parameters=model.named_parameters(),
            op=hvd.Adasum if args.use_adasum else hvd.Average,
            compression=compression,
        )  # device_dense='/cpu:0'

    if cfg.SOLVER.CHECKPOINT_BY_EPOCH:
        ckpt_period = cfg.SOLVER.CHECKPOINT_PERIOD * iters_per_epoch
    else:
        ckpt_period = cfg.SOLVER.CHECKPOINT_PERIOD
    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer,
        ckpt_period,
        max_iter=max_iter,
        max_to_keep=cfg.SOLVER.MAX_TO_KEEP,
    )

    # build writers ==============================================
    tbx_event_writer = get_tbx_event_writer(cfg.OUTPUT_DIR, backup=not cfg.get("RESUME", False))
    tbx_writer = tbx_event_writer._writer  # NOTE: we want to write some non-scalar data
    writers = (
        [
            MyCommonMetricPrinter(max_iter),
            MyJSONWriter(osp.join(cfg.OUTPUT_DIR, "metrics.json")),
            tbx_event_writer,
        ]
        if comm.is_main_process()
        else []
    )

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement
    logger.info("Starting training from iteration {}".format(start_iter))
    iter_time = None
    with EventStorage(start_iter) as storage:
        for iteration in range(start_iter, max_iter):
            storage.iter = iteration
            epoch = iteration // iters_per_epoch + 1  # epoch start from 1
            storage.put_scalar("epoch", epoch, smoothing_hint=False)

            max_refine_iter = max(1, net_cfg.N_ITER_TRAIN)
            if net_cfg.N_ITER_TRAIN_WARM_EPOCH > 0:
                max_refine_iter = min(
                    max_refine_iter,
                    max(1, int(max_refine_iter * epoch / max(net_cfg.N_ITER_TRAIN_WARM_EPOCH, 1))),
                )

            if np.random.rand() < train_2_ratio:
                data = next(data_loader_2_iter)
            else:
                data = next(data_loader_iter)

            if iter_time is not None:
                storage.put_scalar("time", time.perf_counter() - iter_time)
            iter_time = time.perf_counter()

            batch = batch_data(cfg, data)
            poses_est = None
            # refine loop -----------------------------
            for refine_i in range(1, max_refine_iter + 1):
                # batch update ---------
                batch_updater(cfg, batch, renderer, ren_models=ren_models, poses_est=poses_est)
                if cfg.TRAIN.VIS:
                    vis_batch(cfg, batch, phase="train", refine_iter=refine_i)
                # forward ============================================================
                with autocast(enabled=AMP_ON):
                    out_dict, loss_dict = model(
                        batch["zoom_x"] if "zoom_x" in batch else batch["zoom_x_obs"],
                        x_ren=batch.get("zoom_x_ren", None),
                        K_zoom=batch["zoom_K"],
                        init_pose=batch["obj_pose_est"],
                        obj_class=batch["obj_cls"],
                        gt_mask_trunc=batch.get("zoom_trunc_mask", None),
                        gt_mask_visib=batch.get("zoom_visib_mask", None),
                        # gt_mask_full=batch["zoom_full_mask"],
                        gt_ego_rot=batch["obj_pose"][:, :3, :3],
                        gt_trans=batch["obj_pose"][:, :3, 3],
                        gt_points=batch.get("obj_points", None),
                        obj_extent=batch.get("obj_extent", None),
                        sym_info=batch.get("sym_info", None),
                        gt_flow=batch.get("zoom_flow", None),
                        # roi_coord_2d=batch.get("roi_coord_2d", None),
                        do_loss=True,
                        cur_iter=refine_i,
                    )
                    losses = sum(loss_dict.values())
                    assert torch.isfinite(losses).all(), loss_dict

                poses_est = out_dict[f"pose_{refine_i}"].detach()  # gradient not needed

                loss_dict_reduced = {f"iter{refine_i}/{k}": v.item() for k, v in comm.reduce_dict(loss_dict).items()}
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                if comm.is_main_process():
                    storage.put_scalars(**{f"iter{refine_i}/total_loss": losses_reduced})
                    storage.put_scalars(**loss_dict_reduced)

                optimizer.zero_grad()
                if AMP_ON:
                    grad_scaler.scale(losses).backward()

                    # # Unscales the gradients of optimizer's assigned params in-place
                    # grad_scaler.unscale_(optimizer)
                    # # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    if comm._USE_HVD:
                        optimizer.synchronize()
                        with optimizer.skip_synchronize():
                            grad_scaler.step(optimizer)
                            grad_scaler.update()
                    else:
                        grad_scaler.step(optimizer)
                        grad_scaler.update()
                else:
                    losses.backward()
                    optimizer.step()
            # end of refine loop ------------------------------------------------------------------------------------
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if cfg.TEST.EVAL_PERIOD > 0 and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0 and iteration != max_iter - 1:
                do_test(cfg, model, epoch=epoch, iteration=iteration)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            if iteration - start_iter > 5 and (
                (iteration + 1) % cfg.TRAIN.PRINT_FREQ == 0 or iteration == max_iter - 1 or iteration < 100
            ):
                for writer in writers:
                    writer.write()
                # visualize some images ========================================
                if cfg.TRAIN.VIS_IMG:
                    with torch.no_grad():
                        vis_i = 0
                        roi_img_vis = batch["roi_img"][vis_i].cpu().numpy()
                        roi_img_vis = denormalize_image(roi_img_vis, cfg).transpose(1, 2, 0).astype("uint8")
                        tbx_writer.add_image("input_image", roi_img_vis, iteration)

                        out_mask = out_dict["mask"].detach()
                        out_mask = get_out_mask(cfg, out_mask)
                        out_mask_vis = out_mask[vis_i, 0].cpu().numpy()
                        tbx_writer.add_image("out_mask", out_mask_vis, iteration)

                        gt_mask_vis = batch["roi_mask"][vis_i].detach().cpu().numpy()
                        tbx_writer.add_image("gt_mask", gt_mask_vis, iteration)
            periodic_checkpointer.step(iteration, epoch=epoch)
