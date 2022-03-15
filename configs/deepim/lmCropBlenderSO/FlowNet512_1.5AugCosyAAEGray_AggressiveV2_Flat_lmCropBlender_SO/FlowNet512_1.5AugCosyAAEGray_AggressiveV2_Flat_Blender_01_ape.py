_base_ = ["../../../_base_/deepim_base.py"]

OUTPUT_DIR = "output/deepim/lmCropBlenderSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmCropBlender_SO/ape"
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
    INIT_POSE_TYPE_TRAIN=["gt_noise"],  # gt_noise | random | canonical
    NOISE_ROT_STD_TRAIN=(15, 10, 5, 2.5, 1.25),  # randomly choose one
    NOISE_TRANS_STD_TRAIN=[(0.01, 0.01, 0.05), (0.01, 0.01, 0.01), (0.005, 0.005, 0.01)],
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
)

DATASETS = dict(
    TRAIN=("lm_blender_ape_train",),
    TEST=("lm_crop_ape_test",),
    INIT_POSE_FILES_TEST=(
        "datasets/BOP_DATASETS/lm/test/init_poses/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e_so_lmCropTest_GdrnPose_with_bbox_crop.json",
    ),
)

MODEL = dict(
    LOAD_DETS_TEST=False,
    LOAD_POSES_TEST=True,
    PIXEL_MEAN=[0.0, 0.0, 0.0],
    PIXEL_STD=[255.0, 255.0, 255.0],
    DEEPIM=dict(
        NAME="DeepIM_FlowNet",  # used module file name (define different model types)
        TASK="refine",  # refine | init | init+refine
        NUM_CLASSES=11,  # only valid for class aware
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
            PM_LW=1.0,
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

# bbnc5
# iter0
# objects  ape    Avg(1)
# ad_2     0.00   0.00
# ad_5     0.00   0.00
# ad_10    0.00   0.00
# rete_2   0.00   0.00
# rete_5   2.82   2.82
# rete_10  15.32  15.32
# re_2     0.81   0.81
# re_5     8.87   8.87
# re_10    30.65  30.65
# te_2     0.00   0.00
# te_5     4.44   4.44
# te_10    18.15  18.15
# proj_2   0.00   0.00
# proj_5   12.50  12.50
# proj_10  40.73  40.73
# re       41.71  41.71
# te       0.26   0.26

# iter4
# objects  ape    Avg(1)
# ad_2     0.00   0.00
# ad_5     6.45   6.45
# ad_10    21.37  21.37
# rete_2   16.94  16.94
# rete_5   55.24  55.24
# rete_10  59.68  59.68
# re_2     17.74  17.74
# re_5     56.05  56.05
# re_10    60.89  60.89
# te_2     48.39  48.39
# te_5     59.68  59.68
# te_10    67.34  67.34
# proj_2   48.39  48.39
# proj_5   63.71  63.71
# proj_10  69.35  69.35
# re       29.69  29.69
# te       0.11   0.11
