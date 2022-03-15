_base_ = ["../../../_base_/deepim_base.py"]


OUTPUT_DIR = "output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_NoiseRandom_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/01_02MasterChefCan"
INPUT = dict(
    COLOR_AUG_PROB=0.8,
    COLOR_AUG_TYPE="code",
    COLOR_AUG_CODE=(
        "Sequential(["
        # Sometimes(0.5, PerspectiveTransform(0.05)),
        # Sometimes(0.5, CropAndPad(percent=(-0.05, 0.1))),
        # Sometimes(0.5, Affine(scale=(1.0, 1.2))),
        "Sometimes(0.5, CoarseDropout( p=0.2, size_percent=0.05) ),"
        "Sometimes(0.4, GaussianBlur((0., 3.))),"
        "Sometimes(0.3, pillike.EnhanceSharpness(factor=(0., 50.))),"
        "Sometimes(0.3, pillike.EnhanceContrast(factor=(0.2, 50.))),"
        "Sometimes(0.5, pillike.EnhanceBrightness(factor=(0.1, 6.))),"
        "Sometimes(0.3, pillike.EnhanceColor(factor=(0., 20.))),"
        "Sometimes(0.5, Add((-25, 25), per_channel=0.3)),"
        "Sometimes(0.3, Invert(0.2, per_channel=True)),"
        "Sometimes(0.5, Multiply((0.6, 1.4), per_channel=0.5)),"
        "Sometimes(0.5, Multiply((0.6, 1.4))),"
        "Sometimes(0.1, AdditiveGaussianNoise(scale=10, per_channel=True)),"
        "Sometimes(0.5, iaa.contrast.LinearContrast((0.5, 2.2), per_channel=0.3)),"
        "Sometimes(0.5, Grayscale(alpha=(0.0, 1.0))),"  # maybe remove for det
        "], random_order=True)"
        # cosy+aae
    ),
    ZOOM_ENLARGE_SCALE=1.5,
    BBOX_TYPE_TEST="from_pose",  # from_pose | est | gt | gt_aug (TODO)
    INIT_POSE_TYPE_TRAIN=["gt_noise", "random"],  # gt_noise | random | canonical
    NOISE_ROT_STD_TRAIN=(15, 10, 5, 2.5, 1.25),  # randomly choose one
    NOISE_ROT_MAX_TRAIN=None,  # NOTE
    NOISE_TRANS_STD_TRAIN=(0.01, 0.01, 0.05),
    INIT_POSE_TYPE_TEST="est",  # gt_noise | est | canonical
)

SOLVER = dict(
    IMS_PER_BATCH=32,
    TOTAL_EPOCHS=80,
    LR_SCHEDULER_NAME="flat_and_anneal",
    ANNEAL_METHOD="cosine",  # "cosine"
    ANNEAL_POINT=0.72,
    # REL_STEPS=(0.3125, 0.625, 0.9375),
    OPTIMIZER_CFG=dict(_delete_=True, type="Ranger", lr=1e-4, weight_decay=0),
    WEIGHT_DECAY=0.0,
    WARMUP_FACTOR=0.001,
    WARMUP_ITERS=1000,
    CLIP_GRADIENTS=dict(
        ENABLED=True,
        CLIP_TYPE="full_model",
        CLIP_VALUE=1.0,
        NORM_TYPE=2.0,
    ),
)

DATASETS = dict(
    TRAIN=("ycbv_002_master_chef_can_train_pbr",),
    TEST=("ycbv_bop_test",),
    INIT_POSE_FILES_TEST=(
        "datasets/BOP_DATASETS/ycbv/test/init_poses/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_ycbv_test_GdrnPose_with_yolov4_pbr_bbox.json",
    ),
    INIT_POSE_THR=0.3,
    DET_THR=0.3,
    SYM_OBJS=[
        "002_master_chef_can",
        "024_bowl",
        "025_mug",
        "036_wood_block",
        "040_large_marker",
        "051_large_clamp",
        "052_extra_large_clamp",
        "061_foam_brick",
    ],  # ycbv_bop
)

DATALOADER = dict(
    # Number of data loading threads
    NUM_WORKERS=4,
    FILTER_VISIB_THR=0.1,
    FILTER_EMPTY_DETS=True,  # filter images with empty detections
)

MODEL = dict(
    LOAD_DETS_TEST=False,
    LOAD_POSES_TEST=True,
    PIXEL_MEAN=[0.0, 0.0, 0.0],
    PIXEL_STD=[255.0, 255.0, 255.0],
    DEEPIM=dict(
        NAME="DeepIM_FlowNet",  # used module file name (define different model types)
        TASK="refine",  # refine | init | init+refine
        NUM_CLASSES=21,  # only valid for class aware
        N_ITER_TRAIN=4,
        N_ITER_TRAIN_WARM_EPOCH=4,  # linearly increase the refine iter from 1 to N_ITER_TRAIN until this epoch
        N_ITER_TEST=4,
        ## backbone
        BACKBONE=dict(
            PRETRAINED="pretrained_models/flownet/flownets_EPE1.951.pth.tar",
            INIT_CFG=dict(
                type="FlowNetS",
                # [im_ren, im_obs]
                in_channels=6,
                use_bn=False,
                out_flow_level="flow4",
                out_concat4=True,
            ),
            INPUT_H=512,  # use squared image to easily combined with gdrn
            INPUT_W=512,
            INPUT_MASK=False,
        ),
        FLAT_OP="flatten",
        ## pose head for delta R/T
        POSE_HEAD=dict(
            ROT_TYPE="ego_rot6d",  # {ego|allo}_{quat|rot6d}
            INIT_CFG=dict(
                type="FC_RotTransHead",
                in_dim=1024 * 8 * 8,  # should match FLAT_OP
                num_layers=2,
                feat_dim=256,
                norm="GN",  # BN | GN | none
                num_gn_groups=32,
                act="gelu",  # relu | lrelu | silu (swish) | gelu | mish
            ),
            DELTA_T_WEIGHT=0.1,  # deepim-pytorch use 0.1 (for the version without K_zoom/zoom_factor)
            # deepim | cosypose (deepim's delta_z=0 means nochange, cosypose uses 1/exp, so 1 means nochamge)
            T_TRANSFORM_K_AWARE=False,  # whether to use zoomed K; deepim False | cosypose True
            DELTA_Z_STYLE="deepim",
        ),
        # mask head
        MASK_HEAD=dict(
            INIT_CFG=dict(
                type="ConvOutHead",
                in_dim=770,
                num_feat_layers=0,  # only output layer
                feat_dim=256,
                feat_kernel_size=3,
                norm="GN",
                num_gn_groups=32,
                act="gelu",
                out_kernel_size=3,
            ),
        ),
        LOSS_CFG=dict(
            # point matching loss ----------------
            PM_LOSS_SYM=True,  # use symmetric PM loss
            PM_NORM_BY_EXTENT=False,  # 1. / extent.max(1, keepdim=True)[0]
            # if False, the trans loss is in point matching loss
            PM_R_ONLY=False,  # only do R loss in PM
            PM_DISENTANGLE_T=False,  # disentangle R/T
            PM_DISENTANGLE_Z=True,  # disentangle R/xy/z
            PM_T_USE_POINTS=True,  #
            PM_LW=10.0,
            # mask loss --------------------
            MASK_CLASS_AWARE=False,
            MASK_LOSS_TYPE="BCE",  # L1 | BCE | CE
            MASK_LOSS_GT="trunc",  # trunc | visib | obj (not supported yet)
            MASK_LW=1.0,
            # flow loss ------------------
            FLOW_LOSS_TYPE="L1",  # L1 | L2
            FLOW_LW=0.1,
        ),
    ),
)

VAL = dict(
    DATASET_NAME="ycbv",
    SPLIT_TYPE="",
    SCRIPT_PATH="lib/pysixd/scripts/eval_pose_results_more.py",
    TARGETS_FILENAME="test_targets_bop19.json",
    ERROR_TYPES="mspd,mssd,vsd,reS,teS,reteS,ad",
    USE_BOP=True,  # whether to use bop toolkit
)
