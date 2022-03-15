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
from det.yolov4.datasets.data_loader import (
    build_yolo_train_loader,
    build_yolo_test_loader,
)
from .engine_utils import batch_data
from .yolov4_coco_evaluation import YOLOv4COCOEvaluator
from .inference import inference_on_dataset
from det.yolov4.yolo_utils.utils import check_img_size
from lib.torch_utils.torch_utils import ModelEMA

logger = logging.getLogger(__name__)


def get_evaluator(cfg, dataset_name, output_folder=None):
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
        evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
    if evaluator_type in ["coco", "coco_panoptic_seg", "coco_bop"]:
        evaluator_list.append(YOLOv4COCOEvaluator(dataset_name, cfg, True, output_folder))
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

    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_test(cfg, model, epoch=None, iteration=None):
    # check img size ------------------------------
    if hasattr(model, "module"):
        max_stride = int(max(model.module.stride))  # grid size (max stride)
    else:
        max_stride = int(max(model.stride))  # grid size (max stride)
    # verify img size are max-stride-multiples
    img_size = check_img_size(cfg.INPUT.MAX_SIZE_TEST, max_stride)

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
        evaluator = get_evaluator(cfg, dataset_name, eval_out_dir)
        data_loader = build_yolo_test_loader(cfg, dataset_name, stride=max_stride, img_size=img_size)
        results_i = inference_on_dataset(cfg, model, data_loader, evaluator, amp_test=cfg.TEST.AMP_TEST)
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


def do_train(cfg, args, model, optimizer, resume=False):
    model.train()

    # check img size ------------------------------
    if hasattr(model, "module"):
        max_stride = int(max(model.module.stride))  # grid size (max stride)
    else:
        max_stride = int(max(model.stride))  # grid size (max stride)
    # verify img size are max-stride-multiples
    img_size = check_img_size(cfg.INPUT.MAX_SIZE_TRAIN, max_stride)

    # some basic settings =========================
    dataset_meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    # data_ref = ref.__dict__[dataset_meta.ref_key]
    # obj_names = dataset_meta.objs

    # load data ===================================
    train_dset_names = cfg.DATASETS.TRAIN
    data_loader = build_yolo_train_loader(cfg, train_dset_names, stride=max_stride, img_size=img_size)
    data_loader_iter = iter(data_loader)

    # load 2nd train dataloader if needed
    train_2_dset_names = cfg.DATASETS.get("TRAIN2", ())
    train_2_ratio = cfg.DATASETS.get("TRAIN2_RATIO", 0.0)
    if train_2_ratio > 0.0 and len(train_2_dset_names) > 0:
        data_loader_2 = build_yolo_train_loader(cfg, train_2_dset_names, stride=max_stride, img_size=img_size)
        data_loader_2_iter = iter(data_loader_2)
    else:
        data_loader_2 = None
        data_loader_2_iter = None

    ims_per_batch = cfg.SOLVER.IMS_PER_BATCH
    if isinstance(data_loader, AspectRatioGroupedDataset):
        dataset_len = len(data_loader.dataset.dataset)
        if data_loader_2 is not None:
            dataset_len += len(data_loader_2.dataset.dataset)
        iters_per_epoch = dataset_len // ims_per_batch
    else:
        dataset_len = len(data_loader.dataset)
        if data_loader_2 is not None:
            dataset_len += len(data_loader_2.dataset)
        iters_per_epoch = dataset_len // ims_per_batch
    max_iter = cfg.SOLVER.TOTAL_EPOCHS * iters_per_epoch
    dprint("ims_per_batch: ", ims_per_batch)
    dprint("dataset length: ", dataset_len)
    dprint("iters per epoch: ", iters_per_epoch)
    dprint("total iters: ", max_iter)

    bs_ref = cfg.SOLVER.get("REFERENCE_BS", 64)  # nominal batch size
    accumulate_iter = max(round(bs_ref / cfg.SOLVER.IMS_PER_BATCH), 1)  # accumulate loss before optimizing
    # NOTE: update lr every accumulate_iter
    scheduler, lr_func = solver_utils.build_lr_scheduler(
        cfg,
        optimizer,
        total_iters=max_iter // accumulate_iter,
        return_function=True,
    )
    n_warm = max(cfg.SOLVER.WARMUP_ITERS, 3 * iters_per_epoch)

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

    # Exponential moving average (NOTE: initialize ema after loading weights)
    if comm.is_main_process() and cfg.MODEL.EMA.ENABLED:
        ema = ModelEMA(model, **cfg.MODEL.EMA.INIT_CFG)
        ema.updates = start_iter // accumulate_iter
        # save the ema model
        checkpointer.model = ema.ema.module if hasattr(ema.ema, "module") else ema.ema
    else:
        ema = None

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
        optimizer.zero_grad()
        for iteration in range(start_iter, max_iter):
            storage.iter = iteration
            epoch = iteration // iters_per_epoch + 1  # epoch start from 1
            storage.put_scalar("epoch", epoch, smoothing_hint=False)

            if np.random.rand() < train_2_ratio:
                data = next(data_loader_2_iter)
            else:
                data = next(data_loader_iter)

            # Warmup --------------------
            if iteration <= n_warm:
                xi = [0, n_warm]  # x interp
                accumulate_iter = max(
                    1,
                    np.interp(iteration, xi, [1, bs_ref / ims_per_batch]).round(),
                )
                warm_factor = cfg.SOLVER.WARMUP_FACTOR
                bias_warm_factor = cfg.SOLVER.BIAS_WARMUP_FACTOR
                for j, p_group in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.001*lr0 to lr0
                    base_lr = p_group["initial_lr"]
                    warm_lr0 = base_lr * bias_warm_factor if j == 2 else base_lr * warm_factor
                    p_group["lr"] = np.interp(iteration, xi, [warm_lr0, base_lr])
                    if "momentum" in p_group and "momentum" in cfg.SOLVER.OPTIMIZER_CFG:
                        p_group["momentum"] = np.interp(
                            iteration,
                            xi,
                            [0.9, cfg.SOLVER.OPTIMIZER_CFG["momentum"]],
                        )

            if iter_time is not None:
                storage.put_scalar("time", time.perf_counter() - iter_time)
            iter_time = time.perf_counter()

            # forward ============================================================
            batch = batch_data(cfg, data)
            with autocast(enabled=AMP_ON):
                out_dict, loss_dict = model(
                    batch["image"], targets=batch["labels"], do_loss=True
                )  # scaled by batch_size
                losses = sum(loss_dict.values()) * comm.get_world_size()
                assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            # backward & optimize ==============================================
            if AMP_ON:
                grad_scaler.scale(losses).backward()

                # # Unscales the gradients of optimizer's assigned params in-place
                # grad_scaler.unscale_(optimizer)
                # # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                # optimize
                if iteration % accumulate_iter == 0:
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
                # optimize
                if iteration % accumulate_iter == 0:
                    optimizer.step()

            if iteration % accumulate_iter == 0:
                optimizer.zero_grad()
                if ema is not None:
                    ema.update(model)
                storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
                scheduler.step()

            if cfg.TEST.EVAL_PERIOD > 0 and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0 and iteration != max_iter - 1:
                if ema is not None:
                    ema.update_attr(model)
                    do_test(
                        cfg,
                        model=ema.ema.module if hasattr(ema.ema, "module") else ema.ema,
                        epoch=epoch,
                        iteration=iteration,
                    )
                else:
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
                        # TODO
            periodic_checkpointer.step(iteration, epoch=epoch)
