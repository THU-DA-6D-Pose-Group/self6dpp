_base_ = "../../_base_/yolov4_base.py"

OUTPUT_DIR = "output/yolov4/ycbv/yolov4x_640_augCosyAAEGray_ranger_ycbv_pbr_ycbv_train"

INPUT = dict(
    # Whether the model needs RGB, YUV, HSV etc.
    FORMAT="BGR",
    MIN_SIZE_TRAIN=(640,),
    MAX_SIZE_TRAIN=640,
    MIN_SIZE_TEST=672,
    MAX_SIZE_TEST=672,
    # mosaic aug
    AUG_MOSAIC=True,
    # color aug
    COLOR_AUG_PROB=0.8,  # coco not use this, but is it useful for coco?
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
    # hsv color aug
    AUG_HSV_PROB=1.0,
    HSV_H=0.015,  # image HSV-Hue augmentation (fraction)
    HSV_S=0.7,  # image HSV-Saturation augmentation (fraction)
    HSV_V=0.4,  # image HSV-Value augmentation (fraction)
    # geometric aug
    RAND_ROTATE_DEG=0.0,  # image rotation (+/- deg)
    RAND_TRANSLATE=0.0,  # image translation (+/- fraction)
    RAND_SCALE=0.5,  # image scale (+/- gain)
    RAND_SHEAR=0.0,  # image shear (+/- deg)
)


SOLVER = dict(
    IMS_PER_BATCH=4,
    TOTAL_EPOCHS=16,
    REFERENCE_BS=-1,
    # will ignore OPTIMIZER_NAME, BASE_LR, MOMENTUM, WEIGHT_DECAY
    OPTIMIZER_CFG=dict(_delete_=True, type="Ranger", lr=1e-4, weight_decay=0),
    #######
    LR_SCHEDULER_NAME="flat_and_anneal",  # WarmupMultiStepLR | flat_and_anneal
    ANNEAL_METHOD="cosine",
    ANNEAL_POINT=0.72,  # 0.72
    TARGET_LR_FACTOR=0.0,
    # checkpoint
    CHECKPOINT_PERIOD=2,
    CHECKPOINT_BY_EPOCH=True,
    MAX_TO_KEEP=5,
)

DATASETS = dict(TRAIN=("ycbv_train_pbr",), TEST=("ycbv_train_real",))


NUM_CLASSES = 21
MODEL = dict(
    # BaiduDisk: https://pan.baidu.com/s/1YZ8ooif5w5F4UzlkhIFjEg  (6hlw)
    WEIGHTS="pretrained_models/yolov4/yolov4x-mish.pth",
    PIXEL_MEAN=[0, 0, 0],  # to [0,1]
    PIXEL_STD=[255.0, 255.0, 255.0],
    YOLO=dict(
        NAME="yolo",  # used module file name
        MODEL_CFG="yolov4x-mish.yaml",
        NUM_CLASSES=NUM_CLASSES,
        IOU_THR_TRAIN=0.20,  # iou training threshold
        ANCHOR_THR=4.0,  # anchor-multiple threshold
        LOSS_CFG=dict(
            CLS_LW=0.5 * NUM_CLASSES / 80,  # scale coco-tuned to current dataset
            CLS_PW=1.0,  # cls BCELoss positive_weight
            GIOU_LW=0.05,
            GIOU_RATIO=1.0,  # giou loss ratio (obj_loss = 1.0 or giou)
            OBJ_LW=1.0,  # obj loss gain (*=img_size/320 if img_size != 320)
            OBJ_PW=1.0,  # obj BCELoss positive_weight
            FOCAL_LOSS_GAMMA=0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
        ),
    ),
    # some d2 keys but not used
    KEYPOINT_ON=False,
    LOAD_PROPOSALS=False,
)

TEST = dict(
    EVAL_PERIOD=0,
    VIS=False,
    HALF_TEST=True,
    PRECISE_BN=dict(ENABLED=False, NUM_ITER=200),
)

# bbnc6
# |   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl   |
# |:------:|:------:|:------:|:-----:|:------:|:------:|
# | 66.929 | 92.937 | 78.683 | 0.466 | 50.729 | 70.105 |
# [0316_200005] det.yolov4.engine.yolov4_coco_evaluation INFO@312: Per-category bbox AP:
# | category            | AP     | category            | AP     | category         | AP     | category            | AP     | category              | AP     |
# |:--------------------|:-------|:--------------------|:-------|:-----------------|:-------|:--------------------|:-------|:----------------------|:-------|
# | 002_master_chef_can | 75.904 | 003_cracker_box     | 77.806 | 004_sugar_box    | 69.877 | 005_tomato_soup_can | 72.469 | 006_mustard_bottle    | 73.724 |
# | 007_tuna_fish_can   | 60.309 | 008_pudding_box     | 68.921 | 009_gelatin_box  | 58.459 | 010_potted_meat_can | 76.340 | 011_banana            | 60.637 |
# | 019_pitcher_base    | 84.900 | 021_bleach_cleanser | 65.831 | 024_bowl         | 62.715 | 025_mug             | 70.158 | 035_power_drill       | 74.875 |
# | 036_wood_block      | 55.675 | 037_scissors        | 62.890 | 040_large_marker | 40.375 | 051_large_clamp     | 72.164 | 052_extra_large_clamp | 59.988 |
# | 061_foam_brick      | 61.496 |                     |        |                  |        |                     |        |                       |        |
# [0316_200005] det.yolov4.engine.yolov4_coco_evaluation INFO@342: Per-category bbox AR:
# | category            | AR     | category            | AR     | category         | AR     | category            | AR     | category              | AR     |
# |:--------------------|:-------|:--------------------|:-------|:-----------------|:-------|:--------------------|:-------|:----------------------|:-------|
# | 002_master_chef_can | 81.758 | 003_cracker_box     | 83.897 | 004_sugar_box    | 75.463 | 005_tomato_soup_can | 79.215 | 006_mustard_bottle    | 79.493 |
# | 007_tuna_fish_can   | 70.194 | 008_pudding_box     | 75.035 | 009_gelatin_box  | 69.440 | 010_potted_meat_can | 81.283 | 011_banana            | 72.346 |
# | 019_pitcher_base    | 89.343 | 021_bleach_cleanser | 76.691 | 024_bowl         | 78.869 | 025_mug             | 76.428 | 035_power_drill       | 82.131 |
# | 036_wood_block      | 67.646 | 037_scissors        | 74.811 | 040_large_marker | 51.667 | 051_large_clamp     | 80.983 | 052_extra_large_clamp | 72.382 |
# | 061_foam_brick      | 71.748 |                     |        |                  |        |                     |        |                       |        |
