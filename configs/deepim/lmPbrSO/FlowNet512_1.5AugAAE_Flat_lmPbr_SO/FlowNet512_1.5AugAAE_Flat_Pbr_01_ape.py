_base_ = ["../../../_base_/deepim_base.py"]


OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugAAE_Flat_lmPbr_SO/ape"
INPUT = dict(
    COLOR_AUG_PROB=0.8,
    COLOR_AUG_TYPE="code",
    COLOR_AUG_CODE=(
        "Sequential(["
        "Sometimes(0.4, CoarseDropout( p=0.1, size_percent=0.05) ),"
        # "Sometimes(0.5, Affine(scale=(1.0, 1.2))),"
        "Sometimes(0.5, GaussianBlur(np.random.rand())),"
        "Sometimes(0.5, Add((-20, 20), per_channel=0.3)),"
        "Sometimes(0.4, Invert(0.20, per_channel=True)),"
        "Sometimes(0.5, Multiply((0.7, 1.4), per_channel=0.8)),"
        "Sometimes(0.5, Multiply((0.7, 1.4))),"
        "Sometimes(0.5, LinearContrast((0.5, 2.0), per_channel=0.3))"
        "], random_order=False)"
    ),
    ZOOM_ENLARGE_SCALE=1.5,
    BBOX_TYPE_TEST="from_pose",  # from_pose | est | gt | gt_aug (TODO)
    INIT_POSE_TYPE_TRAIN=["gt_noise"],  # gt_noise | random | canonical
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
    TRAIN=("lm_pbr_ape_train",),
    TEST=("lm_real_ape_test",),
    # DET_FILES_TEST=("datasets/BOP_DATASETS/lm/test/test_bboxes/bbox_faster_all.json",),
    # INIT_POSE_FILES_TEST=("datasets/BOP_DATASETS/lm/test/init_poses/init_pose_posecnn_lm.json",),
    INIT_POSE_FILES_TEST=(
        "datasets/BOP_DATASETS/lm/test/init_poses/resnest50d_a6_AugCosyAAE_BG05_lm_pbr_100e_so.json",
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
        NUM_CLASSES=13,  # only valid for class aware
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
            )
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

# bbnc7
# objects  ape     Avg(1)                                                                                                                                                                                     │···············································································································
# ad_2     8.95    8.95                                                                                                                                                                                       │···············································································································
# ad_5     57.33   57.33                                                                                                                                                                                      │···············································································································
# ad_10    90.29   90.29                                                                                                                                                                                      │···············································································································
# rete_2   34.19   34.19                                                                                                                                                                                      │···············································································································
# rete_5   98.10   98.10                                                                                                                                                                                      │···············································································································
# rete_10  100.00  100.00                                                                                                                                                                                     │···············································································································
# re_2     34.19   34.19                                                                                                                                                                                      │···············································································································
# re_5     98.10   98.10                                                                                                                                                                                      │···············································································································
# re_10    100.00  100.00                                                                                                                                                                                     │···············································································································
# te_2     99.71   99.71                                                                                                                                                                                      │···············································································································
# te_5     100.00  100.00                                                                                                                                                                                     │···············································································································
# te_10    100.00  100.00                                                                                                                                                                                     │···············································································································
# proj_2   87.90   87.90                                                                                                                                                                                      │···············································································································
# proj_5   98.67   98.67                                                                                                                                                                                      │···············································································································
# proj_10  100.00  100.00                                                                                                                                                                                     │···············································································································
# re       2.45    2.45                                                                                                                                                                                       │···············································································································
# te       0.01    0.01
