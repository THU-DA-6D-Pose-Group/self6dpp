_base_ = "../../_base_/yolov4_base.py"

OUTPUT_DIR = "output/yolov4/lm/yolov4x_640_augCosyAAEGray_ranger_lm_pbr_lm_test"

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

DATASETS = dict(
    TRAIN=("lm_pbr_13_train",),
    TEST=("lm_13_test",),
    # TEST=("lm_13_train",),
    # TEST=("lm_13_all", )
)


NUM_CLASSES = 13
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
    FILTER_SCENE=True,
    HALF_TEST=True,
    PRECISE_BN=dict(ENABLED=False, NUM_ITER=200),
)


# bbnc7
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.866
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.999
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.994
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.797
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.874
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.876
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.893
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.894
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.894
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.809
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.893
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.909
# [0102_091722 det.yolov4.engine.yolov4_coco_evaluation@277]: Evaluation results for bbox:
# |   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
# |:------:|:------:|:------:|:------:|:------:|:------:|
# | 86.568 | 99.880 | 99.364 | 79.689 | 87.441 | 87.585 |
# [0102_091722 det.yolov4.engine.yolov4_coco_evaluation@309]: Per-category bbox AP:
# | category   | AP     | category   | AP     | category   | AP     | category   | AP     | category    | AP     |
# |:-----------|:-------|:-----------|:-------|:-----------|:-------|:-----------|:-------|:------------|:-------|
# | ape        | 84.296 | benchvise  | 89.604 | camera     | 87.913 | can        | 91.055 | cat         | 88.010 |
# | driller    | 91.693 | duck       | 85.962 | eggbox     | 86.974 | glue       | 76.829 | holepuncher | 83.937 |
# | iron       | 84.946 | lamp       | 86.750 | phone      | 87.417 |            |        |             |        |
# [0102_091722 det.yolov4.engine.yolov4_coco_evaluation@339]: Per-category bbox AR:
# | category   | AR     | category   | AR     | category   | AR     | category   | AR     | category    | AR     |
# |:-----------|:-------|:-----------|:-------|:-----------|:-------|:-----------|:-------|:------------|:-------|
# | ape        | 87.629 | benchvise  | 91.794 | camera     | 90.637 | can        | 93.396 | cat         | 90.749 |
# | driller    | 94.004 | duck       | 89.070 | eggbox     | 90.329 | glue       | 80.917 | holepuncher | 87.222 |
# | iron       | 87.242 | lamp       | 89.395 | phone      | 89.433 |            |        |             |        |


# -------------
# lm_13_all
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.865
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.999
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.992
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.818
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.874
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.875
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.893
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.893
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.893
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.829
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.893
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.909
# [0416_220413 det.yolov4.engine.yolov4_coco_evaluation@289]: Evaluation results for bbox:
# |   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
# |:------:|:------:|:------:|:------:|:------:|:------:|
# | 86.494 | 99.896 | 99.195 | 81.829 | 87.383 | 87.504 |
# [0416_220413 det.yolov4.engine.yolov4_coco_evaluation@321]: Per-category bbox AP:
# | category   | AP     | category   | AP     | category   | AP     | category   | AP     | category    | AP     |
# |:-----------|:-------|:-----------|:-------|:-----------|:-------|:-----------|:-------|:------------|:-------|
# | ape        | 84.141 | benchvise  | 89.473 | camera     | 87.651 | can        | 91.055 | cat         | 88.059 |
# | driller    | 91.728 | duck       | 85.908 | eggbox     | 86.811 | glue       | 76.695 | holepuncher | 83.850 |
# | iron       | 85.093 | lamp       | 86.755 | phone      | 87.205 |            |        |             |        |
# [0416_220413 det.yolov4.engine.yolov4_coco_evaluation@351]: Per-category bbox AR:
# | category   | AR     | category   | AR     | category   | AR     | category   | AR     | category    | AR     |
# |:-----------|:-------|:-----------|:-------|:-----------|:-------|:-----------|:-------|:------------|:-------|
# | ape        | 87.589 | benchvise  | 91.763 | camera     | 90.383 | can        | 93.244 | cat         | 90.814 |
# | driller    | 94.108 | duck       | 89.067 | eggbox     | 90.223 | glue       | 80.795 | holepuncher | 86.985 |
# | iron       | 87.509 | lamp       | 89.535 | phone      | 89.324 |            |        |             |        |
# -----------------------
