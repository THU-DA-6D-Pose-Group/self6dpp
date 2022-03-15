_base_ = ["./common_base.py"]
# base config for self6dpp's refiner

# -----------------------------------------------------------------------------
# Input (override common base)
# -----------------------------------------------------------------------------
INPUT = dict(
    # Whether the model needs RGB, YUV, HSV etc.
    FORMAT="BGR",
    MIN_SIZE_TRAIN=(480,),
    MAX_SIZE_TRAIN=640,
    MIN_SIZE_TRAIN_SAMPLING="choice",
    MIN_SIZE_TEST=480,
    MAX_SIZE_TEST=640,
    WITH_DEPTH=False,
    AUG_DEPTH=False,
    # color aug
    COLOR_AUG_PROB=0.0,
    COLOR_AUG_TYPE="ROI10D",
    COLOR_AUG_CODE="",
    COLOR_AUG_SYN_ONLY=False,
    ## bg images
    BG_TYPE="VOC_table",  # VOC_table | coco | VOC | SUN2012
    BG_IMGS_ROOT="datasets/VOCdevkit/VOC2012/",  # "datasets/coco/train2017/"
    NUM_BG_IMGS=10000,
    CHANGE_BG_PROB=0.5,  # prob to change bg of real image
    # truncation fg (randomly replace some side of fg with bg during replace_bg)
    TRUNCATE_FG=False,
    BG_KEEP_ASPECT_RATIO=True,
    ## input bbox type -------------------------------
    BBOX_TYPE_TEST="est",  # from_pose | est | gt | gt_aug (TODO)
    ## bbox aug
    BBOX_AUG_PROB=0.5,  # for train
    BBOX_AUG_TYPE="uniform",  # uniform, truncnorm, none, roi10d
    BBOX_AUG_SCALE_RATIO=0.25,  # wh scale
    BBOX_AUG_SHIFT_RATIO=0.25,  # center shift
    # NOTE: use 1.5 to be consistent with gdrn
    ZOOM_ENLARGE_SCALE=1.4,  # enlarge scale for the common box of obs/ren
    ## initial pose type ----------------------------------
    # gt_noise: using random perturbated pose based on gt
    # est: using external pose estimates
    # canonical: using DeepIM for initial pose prediction (like cosypose coarse+refine)
    # INIT_POSE_TYPE_TRAIN=["gt_noise"],
    INIT_POSE_TYPE_TRAIN=["gt_noise"],  # randomly chosen from ["gt_noise", "random", "canonical"]
    INIT_POSE_TYPE_TEST="est",  # gt_noise | est | canonical
    ## pose init  (gt+noise) ----------------------------------
    NOISE_ROT_STD_TRAIN=(15, 10, 5, 2.5),  # randomly choose one
    NOISE_ROT_STD_TEST=15,  # if use gt pose + noise as random initial poses
    NOISE_ROT_MAX_TRAIN=45,
    NOISE_ROT_MAX_TEST=45,
    # trans
    NOISE_TRANS_STD_TRAIN=(0.01, 0.01, 0.05),
    NOISE_TRANS_STD_TEST=(0.01, 0.01, 0.05),
    INIT_TRANS_MIN_Z=0.1,
    ## pose init (random) -------------------------------------
    RANDOM_TRANS_MIN=[-0.35, -0.35, 0.5],
    RANDOM_TRANS_MAX=[0.35, 0.35, 1.3],
    ## pose init (canonical) ----------------------------------
    # a chain of axis-angle, the last value will be multiplied by pi
    CANONICAL_ROT=[(1, 0, 0, 0.5), (0, 0, 1, -0.7)],
    CANONICAL_TRANS=[0, 0, 1.0],
)


# -----------------------------------------------------------------------------
# base model cfg for refiner in self6dpp
# -----------------------------------------------------------------------------
MODEL = dict(
    DEVICE="cuda",
    WEIGHTS="",
    PIXEL_MEAN=[0, 0, 0],  # to [0,1]
    PIXEL_STD=[255.0, 255.0, 255.0],
    LOAD_DETS_TEST=False,
    LOAD_POSES_TEST=False,  # NOTE: may override loaded test bboxes !!
    # Deep Iterative Matching
    DEEPIM=dict(
        NAME="DeepIM_FlowNet",  # used module file name (define different model types)
        TASK="refine",  # refine | init | init+refine
        NUM_CLASSES=13,  # only valid for class aware
        N_ITER_TRAIN=4,
        N_ITER_TRAIN_WARM_EPOCH=4,  # linearly increase the refine iter from 1 to N_ITER_TRAIN until this epoch
        N_ITER_TEST=4,
        USE_MTL=False,  # uncertainty multi-task weighting
        INPUT_OBS_TYPE="rgb",  # rgb | xyz (TODO) | xyz_pred (TODO)
        INPUT_REN_TYPE="rgb",  # rgb | xyz
        ## backbone
        BACKBONE=dict(
            FREEZE=False,
            PRETRAINED="pretrained_models/flownet/flownets_EPE1.951.pth.tar",
            INIT_CFG=dict(
                type="FlowNetS",
                # basic: [im_ren, im_obs]
                in_channels=6,
                use_bn=False,
                out_flow_level="flow4",  # none | all | flow4
                out_concat4=True,
            ),
            SHARED=True,
            INPUT_MASK=False,  # default True in DeepIM
            INPUT_DEPTH=False,  # TODO: for RGB-D
            INPUT_H=480,
            INPUT_W=640,
        ),
        BACKBONE_REN=dict(
            ENABLED=False,
            FREEZE=False,
            PRETRAINED="timm",
            INIT_CFG=dict(
                type="timm/resnest26d",
                in_chans=3,
                pretrained=True,  # imagenet-1k weights
                # checkpoint_path="",  # this is for custom weights
                features_only=True,
                out_indices=(4,),
            ),
        ),
        ## backbone ren (for unshared version)
        FLAT_OP="avg-max-min",  # flat | avg | avg-max-min | avg-max
        ## pose head for delta R/T
        POSE_HEAD=dict(
            FREEZE=False,
            ROT_TYPE="ego_rot6d",  # {ego|allo}_{quat|rot6d}
            CLASS_AWARE=False,
            INIT_CFG=dict(
                type="FC_RotTransHead",
                in_dim=1024 * 3,  # should match FLAT_OP
                num_layers=2,
                feat_dim=256,
                norm="GN",  # BN | GN | none
                num_gn_groups=32,
                act="gelu",  # relu | lrelu | silu (swish) | gelu | mish
            ),
            LR_MULT=1.0,
            DELTA_T_SPACE="image",  # image | 3D
            DELTA_T_WEIGHT=1.0,  # deepim-pytorch use 0.1 (for the version without K_zoom/zoom_factor)
            # deepim | cosypose (deepim's delta_z=0 means nochange, cosypose uses 1/exp, so 1 means nochamge)
            T_TRANSFORM_K_AWARE=True,  # whether to use zoomed K; deepim False | cosypose True
            DELTA_Z_STYLE="cosypose",
        ),
        # mask head
        MASK_HEAD=dict(
            ENABLED=True,
            FREEZE=False,
            CLASS_AWARE=False,
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
            LR_MULT=1.0,
            MASK_THR_TEST=0.5,
        ),
        # flow_head ----------- (external flow head, not used for flownet)
        FLOW_HEAD=dict(
            ENABLED=False,
            FREEZE=False,
            CLASS_AWARE=False,
            INIT_CFG=dict(
                type="TopDownHead",
                in_dim=384,  # this is num out channels of efficientnet_b3
                up_types=(
                    "bilinear",
                    "bilinear",
                    "bilinear",
                ),  # stride 32 to 4
                deconv_kernel_size=3,
                num_conv_per_block=2,
                feat_dim=256,
                feat_kernel_size=3,
                norm="GN",
                num_gn_groups=32,
                act="GELU",
                out_kernel_size=1,
                out_dim=2,
            ),
            LR_MULT=1.0,
        ),
        LOSS_CFG=dict(
            # point matching loss ----------------
            NUM_PM_POINTS=3000,
            PM_LOSS_TYPE="L1",  # L1 | Smooth_L1
            PM_SMOOTH_L1_BETA=1.0,
            PM_LOSS_SYM=False,  # use symmetric PM loss
            PM_NORM_BY_EXTENT=False,  # 1. / extent.max(1, keepdim=True)[0]
            # if False, the trans loss is in point matching loss
            PM_R_ONLY=False,  # only do R loss in PM
            PM_DISENTANGLE_T=False,  # disentangle R/T
            PM_DISENTANGLE_Z=True,  # disentangle R/xy/z
            PM_T_USE_POINTS=True,  # only used for disentangled loss
            PM_LW=1.0,
            # rot loss (default not enabled)--------------
            ROT_LOSS_TYPE="angular",  # angular | L2
            ROT_LW=0.0,
            # trans loss (default not enabled) -----------
            TRANS_LOSS_TYPE="L1",
            TRANS_LOSS_DISENTANGLE=True,
            TRANS_LW=0.0,
            # mask loss --------------------
            MASK_CLASS_AWARE=False,
            MASK_LOSS_TYPE="BCE",  # L1 | BCE | CE
            MASK_LOSS_GT="trunc",  # trunc | visib | obj (not supported yet)
            MASK_LW=1.0,
            # flow loss ------------------
            FLOW_LOSS_TYPE="L1",  # L1 | L2
            FLOW_LW=0.0,
        ),
    ),
    # some d2 keys but not used
    KEYPOINT_ON=False,
    LOAD_PROPOSALS=False,
)

TEST = dict(
    EVAL_PERIOD=0,
    VIS=False,
    SAVE_RESULTS_ONLY=False,
    OUTPUT_MASK=False,  # whether to output mask
    OUTPUT_FLOW=False,
    PRECISE_BN=dict(ENABLED=False, NUM_ITER=200),  # d2 keys, not used
)
