import logging
import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
import os.path as osp
import sys
from setproctitle import setproctitle
import torch
from torch.nn.parallel import DistributedDataParallel

from detectron2.engine import launch
from detectron2.data import MetadataCatalog
from mmcv import Config
import cv2
from loguru import logger as loguru_logger

cv2.setNumThreads(0)  # pytorch issue 1355: possible deadlock in dataloader
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../"))
from core.utils.default_args_setup import (
    my_default_argument_parser,
    my_default_setup,
)
from core.utils.my_setup import setup_for_distributed
from core.utils.my_checkpoint import MyCheckpointer
from core.utils import my_comm as comm

from lib.utils.utils import iprint
from lib.utils.setup_logger import setup_my_logger
from lib.utils.time_utils import get_time_str
import ref

from core.deepim.datasets.dataset_factory import register_datasets_in_cfg
from core.deepim.engine.engine_utils import get_renderer
from core.deepim.engine.engine import do_test, do_train, do_save_results
from core.deepim.models import DeepIM_FlowNet, DeepIM_Shared, DeepIM_Unshared  # noqa


logger = logging.getLogger("detectron2")


def setup(args):
    """Create configs and perform basic setups."""
    cfg = Config.fromfile(args.config_file)
    if args.opts is not None:
        cfg.merge_from_dict(args.opts)
    ############## pre-process some cfg options ######################
    # NOTE: check if need to set OUTPUT_DIR automatically
    if cfg.OUTPUT_DIR.lower() == "auto":
        cfg.OUTPUT_DIR = osp.join(
            cfg.OUTPUT_ROOT,
            osp.splitext(args.config_file)[0].split("configs/")[1],
        )
        iprint(f"OUTPUT_DIR was automatically set to: {cfg.OUTPUT_DIR}")

    if cfg.get("EXP_NAME", "") == "":
        setproctitle("{}.{}".format(osp.splitext(osp.basename(args.config_file))[0], get_time_str()))
    else:
        setproctitle("{}.{}".format(cfg.EXP_NAME, get_time_str()))

    if cfg.SOLVER.AMP.ENABLED:
        if torch.cuda.get_device_capability() <= (6, 1):
            iprint("Disable AMP for older GPUs")
            cfg.SOLVER.AMP.ENABLED = False

    # NOTE: pop some unwanted configs in detectron2
    # ---------------------------------------------------------
    cfg.SOLVER.pop("STEPS", None)
    cfg.SOLVER.pop("MAX_ITER", None)
    # NOTE: get optimizer from string cfg dict
    if cfg.SOLVER.OPTIMIZER_CFG != "":
        if isinstance(cfg.SOLVER.OPTIMIZER_CFG, str):
            optim_cfg = eval(cfg.SOLVER.OPTIMIZER_CFG)
            cfg.SOLVER.OPTIMIZER_CFG = optim_cfg
        else:
            optim_cfg = cfg.SOLVER.OPTIMIZER_CFG
        iprint("optimizer_cfg:", optim_cfg)
        cfg.SOLVER.OPTIMIZER_NAME = optim_cfg["type"]
        cfg.SOLVER.BASE_LR = optim_cfg["lr"]
        cfg.SOLVER.MOMENTUM = optim_cfg.get("momentum", 0.9)
        cfg.SOLVER.WEIGHT_DECAY = optim_cfg.get("weight_decay", 1e-4)
    if cfg.get("DEBUG", False):
        iprint("DEBUG")
        args.num_gpus = 1
        args.num_machines = 1
        cfg.DATALOADER.NUM_WORKERS = 0
        cfg.TRAIN.PRINT_FREQ = 1
    # register datasets
    register_datasets_in_cfg(cfg)

    exp_id = "{}".format(osp.splitext(osp.basename(args.config_file))[0])

    if args.eval_only:
        exp_id += "_test"
    cfg.EXP_ID = exp_id
    cfg.RESUME = args.resume
    ####################################
    if args.launcher != "none":
        comm.init_dist(args.launcher, **cfg.DIST_PARAMS)
    # cfg.freeze()
    my_default_setup(cfg, args)
    # Setup logger
    setup_my_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="core")
    setup_my_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="lib")
    setup_for_distributed(is_master=comm.is_main_process())
    return cfg


@loguru_logger.catch
def main(args):
    cfg = setup(args)

    distributed = comm.get_world_size() > 1

    # get renderer ----------------------
    train_dset_meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    data_ref = ref.__dict__[train_dset_meta.ref_key]
    train_obj_names = train_dset_meta.objs

    render_gpu_id = comm.get_local_rank()
    renderer = get_renderer(cfg, data_ref, obj_names=train_obj_names, gpu_id=render_gpu_id)

    logger.info(f"Used DeepIM module name: {cfg.MODEL.DEEPIM.NAME}")
    model, optimizer = eval(cfg.MODEL.DEEPIM.NAME).build_model_optimizer(cfg, is_test=args.eval_only)
    logger.info("Model:\n{}".format(model))
    if True:
        # sum(p.numel() for p in model.parameters() if p.requires_grad)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info("{}M params".format(params))

    if cfg.TEST.SAVE_RESULTS_ONLY:  # save results only ------------------------------
        MyCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        return do_save_results(cfg, model, renderer=renderer)

    if args.eval_only:  # eval only --------------------------------------------------
        MyCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        return do_test(cfg, model, renderer=renderer)

    if distributed and args.launcher not in ["hvd"]:
        model = DistributedDataParallel(
            model,
            device_ids=[comm.get_local_rank()],
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    do_train(cfg, args, model, optimizer, renderer=renderer, resume=args.resume)
    return do_test(cfg, model, renderer=renderer)


if __name__ == "__main__":
    import resource

    # RuntimeError: received 0 items of ancdata. Issue: pytorch/pytorch#973
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    hard_limit = rlimit[1]
    soft_limit = min(500000, hard_limit)
    iprint("soft limit: ", soft_limit, "hard limit: ", hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

    args = my_default_argument_parser().parse_args()
    iprint("Command Line Args:", args)
    comm.init_dist_env_variables(args)

    if args.eval_only:
        torch.multiprocessing.set_sharing_strategy("file_system")

    if args.launcher != "none":
        main(args)
    else:
        launch(
            main,
            args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url=args.dist_url,
            args=(args,),
        )
