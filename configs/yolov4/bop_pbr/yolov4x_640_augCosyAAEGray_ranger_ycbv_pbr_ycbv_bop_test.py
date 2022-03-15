_base_ = "../../_base_/yolov4_base.py"

OUTPUT_DIR = "output/yolov4/ycbv/yolov4x_640_augCosyAAEGray_ranger_ycbv_pbr_ycbv_test"

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

DATASETS = dict(TRAIN=("ycbv_train_pbr",), TEST=("ycbv_bop_test",))


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

# gu
# [0120_145618 det.yolov4.engine.inference@114]: Total inference time: 0:00:23.180578 (0.025900 s / img per device, on 1 devices)
# [0120_145618 det.yolov4.engine.inference@122]: Total inference pure compute time: 0:00:20 (0.023290 s / img per device, on 1 devices)
# [0120_145618 det.yolov4.engine.inference@127]: Total inference nms time: 0:00:01 (0.001137 s / img per device, on 1 devices)
# [0120_145618 det.yolov4.engine.inference@132]: Total inference compute+nms time: 0:00:21 (0.024426 s / img per device, on 1 devices)
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.780
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.938
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.907
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.056
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.677
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.825
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.815
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.844
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.850
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.525
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.773
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.878
# [0120_145621 det.yolov4.engine.yolov4_coco_evaluation@280]: Evaluation results for bbox:
# |   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl   |
# |:------:|:------:|:------:|:-----:|:------:|:------:|
# | 77.954 | 93.789 | 90.714 | 5.639 | 67.709 | 82.531 |
# [0120_145621 det.yolov4.engine.yolov4_coco_evaluation@312]: Per-category bbox AP:
# | category            | AP     | category            | AP     | category         | AP     | category            | AP     | category              | AP     |
# |:--------------------|:-------|:--------------------|:-------|:-----------------|:-------|:--------------------|:-------|:----------------------|:-------|
# | 002_master_chef_can | 80.806 | 003_cracker_box     | 88.052 | 004_sugar_box    | 87.064 | 005_tomato_soup_can | 78.321 | 006_mustard_bottle    | 88.969 |
# | 007_tuna_fish_can   | 84.440 | 008_pudding_box     | 89.361 | 009_gelatin_box  | 88.765 | 010_potted_meat_can | 75.626 | 011_banana            | 78.857 |
# | 019_pitcher_base    | 92.717 | 021_bleach_cleanser | 74.852 | 024_bowl         | 83.723 | 025_mug             | 85.475 | 035_power_drill       | 87.812 |
# | 036_wood_block      | 70.430 | 037_scissors        | 18.085 | 040_large_marker | 81.542 | 051_large_clamp     | 62.141 | 052_extra_large_clamp | 56.161 |
# | 061_foam_brick      | 83.834 |                     |        |                  |        |                     |        |                       |        |
# [0120_145621 det.yolov4.engine.yolov4_coco_evaluation@342]: Per-category bbox AR:
# | category            | AR     | category            | AR     | category         | AR     | category            | AR     | category              | AR     |
# |:--------------------|:-------|:--------------------|:-------|:-----------------|:-------|:--------------------|:-------|:----------------------|:-------|
# | 002_master_chef_can | 83.067 | 003_cracker_box     | 90.489 | 004_sugar_box    | 89.547 | 005_tomato_soup_can | 82.578 | 006_mustard_bottle    | 90.000 |
# | 007_tuna_fish_can   | 87.467 | 008_pudding_box     | 90.800 | 009_gelatin_box  | 90.267 | 010_potted_meat_can | 82.756 | 011_banana            | 83.733 |
# | 019_pitcher_base    | 94.889 | 021_bleach_cleanser | 80.633 | 024_bowl         | 88.533 | 025_mug             | 87.467 | 035_power_drill       | 88.733 |
# | 036_wood_block      | 73.333 | 037_scissors        | 68.400 | 040_large_marker | 83.267 | 051_large_clamp     | 81.400 | 052_extra_large_clamp | 79.667 |
# | 061_foam_brick      | 87.067 |                     |        |                  |        |                     |        |                       |        |
