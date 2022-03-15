import datetime
import logging
import time
from collections import abc
import torch
from torch.cuda.amp import autocast
from core.utils.my_comm import get_world_size
from detectron2.utils.logger import log_every_n_seconds
from detectron2.evaluation import inference_context, DatasetEvaluators
from det.yolov4.engine.engine_utils import batch_data
from det.yolov4.yolo_utils.utils import non_max_suppression


def inference_on_dataset(cfg, model, data_loader, evaluator, amp_test=False, device="cuda"):
    """#NOTE: modified to add time
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.forward` accurately.
    The model will be used in eval mode.
    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    net_cfg = cfg.MODEL.YOLO
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    half_test = cfg.TEST.HALF_TEST
    if half_test:
        logger.info("half test")
        amp_test = False
        model.half()
    if amp_test:
        logger.info("amp test")

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    total_nms_time = 0
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == 0:  # warmup
                for _ in range(5):
                    batch = batch_data(cfg, inputs, phase="test")
                    if half_test:
                        _ = model(batch["image"].half())
                    else:
                        _ = model(batch["image"])
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
                total_nms_time = 0

            start_compute_time = time.perf_counter()
            #############################
            # process input
            batch = batch_data(cfg, inputs, phase="test")
            _, _, height, width = batch["image"].shape  # batch size, channels, height, width
            whwh = torch.Tensor([width, height, width, height]).to(device)
            out_dict = {"height": height, "width": width, "whwh": whwh}
            if half_test:
                inf_out, train_out = model(batch["image"].half())
            else:
                with autocast(enabled=amp_test):
                    inf_out, train_out = model(batch["image"])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            cur_compute_time = time.perf_counter() - start_compute_time
            total_compute_time += cur_compute_time

            # Run NMS
            start_nms_time = time.perf_counter()
            nms_out = non_max_suppression(
                inf_out,
                conf_thres=net_cfg.CONF_THR_TEST,
                iou_thres=net_cfg.IOU_THR_TEST,
                merge=net_cfg.USE_MERGE_NMS,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            cur_nms_time = time.perf_counter() - start_nms_time
            total_nms_time += cur_nms_time
            out_dict["time"] = cur_compute_time + cur_nms_time
            out_dict["nms_out"] = nms_out
            evaluator.process(inputs, out_dict)

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = (total_compute_time + total_nms_time) / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(idx + 1, total, seconds_per_img, str(eta)),
                    n=5,
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    total_nms_time_str = str(datetime.timedelta(seconds=int(total_nms_time)))
    total_fwd_time_str = str(datetime.timedelta(seconds=int(total_compute_time + total_nms_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str,
            total_compute_time / (total - num_warmup),
            num_devices,
        )
    )
    logger.info(
        "Total inference nms time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_nms_time_str,
            total_nms_time / (total - num_warmup),
            num_devices,
        )
    )
    logger.info(
        "Total inference compute+nms time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_fwd_time_str,
            (total_compute_time + total_nms_time) / (total - num_warmup),
            num_devices,
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results
