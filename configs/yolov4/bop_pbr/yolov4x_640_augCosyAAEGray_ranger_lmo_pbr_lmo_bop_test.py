_base_ = "../../_base_/yolov4_base.py"

OUTPUT_DIR = "output/yolov4/lmo/yolov4x_640_augCosyAAEGray_ranger_lmo_pbr_lmo_test"

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

DATASETS = dict(TRAIN=("lmo_pbr_train",), TEST=("lmo_bop_test",))


NUM_CLASSES = 8
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

# bbnc7
# [0120_143621 det.yolov4.engine.inference@114]: Total inference time: 0:00:04.831533 (0.024777 s / img per device, on 1 devices)
# [0120_143621 det.yolov4.engine.inference@122]: Total inference pure compute time: 0:00:04 (0.021486 s / img per device, on 1 devices)
# [0120_143621 det.yolov4.engine.inference@127]: Total inference nms time: 0:00:00 (0.000908 s / img per device, on 1 devices)
# [0120_143621 det.yolov4.engine.inference@132]: Total inference compute+nms time: 0:00:04 (0.022394 s / img per device, on 1 devices)
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.643
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.897
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.742
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.349
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.695
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.726
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.679
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.717
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.727
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.446
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.765
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.772
# [0120_143621 det.yolov4.engine.yolov4_coco_evaluation@280]: Evaluation results for bbox:
# |   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
# |:------:|:------:|:------:|:------:|:------:|:------:|
# | 64.277 | 89.708 | 74.240 | 34.913 | 69.466 | 72.618 |
# [0120_143621 det.yolov4.engine.yolov4_coco_evaluation@312]: Per-category bbox AP:
# | category   | AP     | category   | AP     | category    | AP     | category   | AP     | category   | AP     |
# |:-----------|:-------|:-----------|:-------|:------------|:-------|:-----------|:-------|:-----------|:-------|
# | ape        | 57.987 | can        | 81.140 | cat         | 61.313 | driller    | 75.553 | duck       | 72.366 |
# | eggbox     | 34.971 | glue       | 54.787 | holepuncher | 76.102 |            |        |            |        |
# [0120_143621 det.yolov4.engine.yolov4_coco_evaluation@342]: Per-category bbox AR:
# | category   | AR     | category   | AR     | category    | AR     | category   | AR     | category   | AR     |
# |:-----------|:-------|:-----------|:-------|:------------|:-------|:-----------|:-------|:-----------|:-------|
# | ape        | 67.433 | can        | 84.925 | cat         | 71.633 | driller    | 82.600 | duck       | 77.766 |
# | eggbox     | 54.526 | glue       | 61.818 | holepuncher | 81.150 |            |        |            |        |
