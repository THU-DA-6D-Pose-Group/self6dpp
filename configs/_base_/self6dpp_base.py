_base_ = ["./common_base.py", "./renderer_base.py"]
# -----------------------------------------------------------------------------
# base model cfg for self6d-v2
# -----------------------------------------------------------------------------

refiner_cfg_path = "configs/_base_/self6dpp_refiner_base.py"

MODEL = dict(
    DEVICE="cuda",
    WEIGHTS="",
    REFINER_WEIGHTS="",
    PIXEL_MEAN=[0, 0, 0],  # to [0,1]
    PIXEL_STD=[255.0, 255.0, 255.0],
    SELF_TRAIN=False,  # whether to do self-supervised training
    FREEZE_BN=False,  # use frozen_bn for self-supervised training
    WITH_REFINER=False,  # whether to use refiner
    # -----------
    LOAD_DETS_TRAIN=False,  # NOTE: load detections for self-train
    LOAD_DETS_TRAIN_WITH_POSE=False,  # load detections with pose_refine as pseudo pose
    PSEUDO_POSE_TYPE="pose_refine",  # pose_est | pose_refine | pose_init (online inferred by teacher)
    LOAD_DETS_TEST=False,
    BBOX_CROP_REAL=False,  # whether to use bbox_128, for cropped lm
    BBOX_CROP_SYN=False,
    # -----------
    # Model Exponential Moving Average https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    # NOTE: momentum-based mean teacher
    EMA=dict(
        ENABLED=False,
        INIT_CFG=dict(decay=0.999, updates=0),  # epoch-based
        UPDATE_FREQ=10,  # update the mean teacher every n epochs
    ),
    POSE_NET=dict(
        NAME="GDRN",  # used module file name
        # NOTE: for self-supervised training phase, use offline labels should be more accurate
        XYZ_ONLINE=False,  # rendering xyz online
        XYZ_BP=True,  # calculate xyz from depth by backprojection
        NUM_CLASSES=13,
        USE_MTL=False,  # uncertainty multi-task weighting, TODO: implement for self loss
        INPUT_RES=256,
        OUTPUT_RES=64,
        ## backbone
        BACKBONE=dict(
            FREEZE=False,
            PRETRAINED="timm",
            INIT_CFG=dict(
                type="timm/resnet34",
                pretrained=True,
                in_chans=3,
                features_only=True,
                out_indices=(4,),
            ),
        ),
        NECK=dict(
            ENABLED=False,
            FREEZE=False,
            LR_MULT=1.0,
            INIT_CFG=dict(
                type="FPN",
                in_channels=[256, 512, 1024, 2048],
                out_channels=256,
                num_outs=4,
            ),
        ),
        ## geo head: Mask, XYZ, Region
        GEO_HEAD=dict(
            FREEZE=False,
            LR_MULT=1.0,
            INIT_CFG=dict(
                type="TopDownMaskXyzRegionHead",
                in_dim=512,  # this is num out channels of backbone conv feature
                up_types=("deconv", "bilinear", "bilinear"),  # stride 32 to 4
                deconv_kernel_size=3,
                num_conv_per_block=2,
                feat_dim=256,
                feat_kernel_size=3,
                norm="GN",
                num_gn_groups=32,
                act="GELU",  # relu | lrelu | silu (swish) | gelu | mish
                out_kernel_size=1,
                out_layer_shared=True,
            ),
            XYZ_BIN=64,  # for classification xyz, the last one is bg
            XYZ_CLASS_AWARE=False,
            MASK_CLASS_AWARE=False,
            REGION_CLASS_AWARE=False,
            MASK_THR_TEST=0.5,
            # for region classification, 0 is bg, [1, num_regions]
            # num_regions <= 1: no region classification
            NUM_REGIONS=64,
        ),
        ## for direct regression
        PNP_NET=dict(
            FREEZE=False,
            TRAIN_R_ONLY=False,  # only train fc_r (only valid when FREEZE=False)
            LR_MULT=1.0,
            # ConvPnPNet | SimplePointPnPNet | PointPnPNet | ResPointPnPNet
            INIT_CFG=dict(
                type="ConvPnPNet",
                norm="GN",
                act="relu",
                num_gn_groups=32,
                drop_prob=0.0,  # 0.25
                denormalize_by_extent=True,
            ),
            WITH_2D_COORD=False,  # using 2D XY coords
            COORD_2D_TYPE="abs",  # rel | abs
            REGION_ATTENTION=False,  # region attention
            MASK_ATTENTION="none",  # none | concat | mul
            ROT_TYPE="ego_rot6d",  # {allo/ego}_{quat/rot6d/log_quat/lie_vec}
            TRANS_TYPE="centroid_z",  # trans | centroid_z (SITE) | centroid_z_abs
            Z_TYPE="REL",  # REL | ABS | LOG | NEG_LOG  (only valid for centroid_z)
        ),
        LOSS_CFG=dict(
            # xyz loss ----------------------------
            XYZ_LOSS_TYPE="L1",  # L1 | CE_coor
            XYZ_LOSS_MASK_GT="visib",  # trunc | visib | obj
            XYZ_LW=1.0,
            # full mask loss ---------------------------
            FULL_MASK_LOSS_TYPE="BCE",  # L1 | BCE | CE
            FULL_MASK_LW=0.0,
            # mask loss ---------------------------
            MASK_LOSS_TYPE="L1",  # L1 | BCE | CE | RW_BCE | dice
            MASK_LOSS_GT="trunc",  # trunc | visib | gt
            MASK_LW=1.0,
            # region loss -------------------------
            REGION_LOSS_TYPE="CE",  # CE
            REGION_LOSS_MASK_GT="visib",  # trunc | visib | obj
            REGION_LW=1.0,
            # point matching loss -----------------
            NUM_PM_POINTS=3000,
            PM_LOSS_TYPE="L1",  # L1 | Smooth_L1
            PM_SMOOTH_L1_BETA=1.0,
            PM_LOSS_SYM=False,  # use symmetric PM loss
            PM_NORM_BY_EXTENT=False,  # 10. / extent.max(1, keepdim=True)[0]
            # if False, the trans loss is in point matching loss
            PM_R_ONLY=True,  # only do R loss in PM
            PM_DISENTANGLE_T=False,  # disentangle R/T
            PM_DISENTANGLE_Z=False,  # disentangle R/xy/z
            PM_T_USE_POINTS=True,
            PM_LW=1.0,
            # rot loss ----------------------------
            ROT_LOSS_TYPE="angular",  # angular | L2
            ROT_LW=0.0,
            # centroid loss -----------------------
            CENTROID_LOSS_TYPE="L1",
            CENTROID_LW=1.0,
            # z loss ------------------------------
            Z_LOSS_TYPE="L1",
            Z_LW=1.0,
            # trans loss --------------------------
            TRANS_LOSS_TYPE="L1",
            TRANS_LOSS_DISENTANGLE=True,
            TRANS_LW=0.0,
            # bind term loss: R^T@t ---------------
            BIND_LOSS_TYPE="L1",
            BIND_LW=0.0,
        ),
        SELF_LOSS_CFG=dict(
            # LAB space loss ------------------
            LAB_NO_L=True,
            LAB_LW=0.0,
            # MS-SSIM loss --------------------
            MS_SSIM_LW=0.0,
            # perceptual loss -----------------
            # PERCEPT_CFG=
            PERCEPT_LW=0.0,
            # mask loss (init, ren) -----------------------
            MASK_WEIGHT_TYPE="edge_lower",  # none | edge_lower | edge_higher
            MASK_INIT_REN_LOSS_TYPE="RW_BCE",  # L1 | RW_BCE (re-weighted BCE) | dice
            MASK_INIT_REN_LW=1.0,
            # depth-based geometric loss ------
            GEOM_LOSS_TYPE="chamfer",  # L1, chamfer
            GEOM_LW=0.0,  # 100
            CHAMFER_CENTER_LW=0.0,
            CHAMFER_DIST_THR=0.5,
            # refiner-based loss --------------
            REFINE_LW=0.0,
            # xyz loss (init, ren)
            XYZ_INIT_REN_LOSS_TYPE="L1",  # L1 | CE_coor (for cls)
            XYZ_INIT_REN_LW=0.0,
            # xyz loss (init, pred)
            XYZ_INIT_PRED_LOSS_TYPE="L1",  # L1 | smoothL1
            XYZ_INIT_PRED_LW=0.0,
            # region loss
            REGION_INIT_PRED_LW=0.0,
            # losses between init and pred ==========================
            # mask loss (init, pred) -----------------------
            MASK_TYPE="vis",  # vis | full
            MASK_INIT_PRED_LOSS_TYPE="RW_BCE",  # L1 | RW_BCE (re-weighted BCE)
            MASK_INIT_PRED_LW=0.0,
            MASK_INIT_PRED_TYPE=("vis",),  # ("vis","full",)
            # point matching loss using pseudo pose ---------------------------
            SELF_PM_CFG=dict(
                loss_type="L1",
                beta=1.0,
                reduction="mean",
                loss_weight=0.0,  # NOTE: >0 to enable this loss
                norm_by_extent=False,
                symmetric=True,
                disentangle_t=True,
                disentangle_z=True,
                t_loss_use_points=True,
                r_only=False,
            ),
        ),
    ),
    # some d2 keys but not used
    KEYPOINT_ON=False,
    LOAD_PROPOSALS=False,
)

TRAIN = dict(PRINT_FREQ=20, DEBUG_SINGLE_IM=False)

TEST = dict(
    EVAL_PERIOD=0,
    VIS=False,
    TEST_BBOX_TYPE="est",  # gt | est
    USE_PNP=False,  # use pnp or direct prediction
    SAVE_RESULTS_ONLY=False,  # turn this on to only save the predicted results
    # ransac_pnp | net_iter_pnp (learned pnp init + iter pnp) | net_ransac_pnp (net init + ransac pnp)
    # net_ransac_pnp_rot (net_init + ransanc pnp --> net t + pnp R)
    PNP_TYPE="ransac_pnp",
    PRECISE_BN=dict(ENABLED=False, NUM_ITER=200),
)
