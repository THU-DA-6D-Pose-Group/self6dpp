_base_ = "./yolo_common_base.py"

# -----------------------------------------------------------------------------
# Input (redefine input cfg for yolov4)
# -----------------------------------------------------------------------------
INPUT = dict(
    # Whether the model needs RGB, YUV, HSV etc.
    # NOTE: yolo's dataloader use BGR, but later convert to RGB for model
    FORMAT="BGR",
    MIN_SIZE_TRAIN=(640,),
    MAX_SIZE_TRAIN=640,
    MIN_SIZE_TRAIN_SAMPLING="choice",
    MIN_SIZE_TEST=672,
    MAX_SIZE_TEST=672,
    WITH_DEPTH=False,
    AUG_DEPTH=False,
    # mosaic aug
    AUG_MOSAIC=False,
    # color aug
    COLOR_AUG_PROB=0.0,  # coco not use this, but is it useful for coco?
    COLOR_AUG_TYPE="ROI10D",
    COLOR_AUG_CODE="",
    COLOR_AUG_SYN_ONLY=False,
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
    # flip
    RAND_HFLIP=True,
    RAND_VFLIP=False,
    ## bg images
    BG_TYPE="VOC_table",  # VOC_table | coco | VOC | SUN2012
    BG_IMGS_ROOT="datasets/VOCdevkit/VOC2012/",  # "datasets/coco/train2017/"
    NUM_BG_IMGS=10000,
    CHANGE_BG_PROB=0.5,  # prob to change bg of real image
    # truncation fg (randomly replace some side of fg with bg during replace_bg)
    TRUNCATE_FG=False,
    BG_KEEP_ASPECT_RATIO=True,
)

# -----------------------------------------------------------------------------
# base model cfg for yolov4
# -----------------------------------------------------------------------------
NUM_CLASSES = 13
MODEL = dict(
    DEVICE="cuda",
    WEIGHTS="",
    # PIXEL_MEAN = [103.530, 116.280, 123.675]  # bgr
    # PIXEL_STD = [57.375, 57.120, 58.395]
    # PIXEL_MEAN = [123.675, 116.280, 103.530]  # rgb
    # PIXEL_STD = [58.395, 57.120, 57.375]
    PIXEL_MEAN=[0, 0, 0],  # to [0,1]
    PIXEL_STD=[255.0, 255.0, 255.0],
    # Model Exponential Moving Average https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    EMA=dict(ENABLED=True, INIT_CFG=dict(decay=0.9999, updates=0)),
    YOLO=dict(
        NAME="yolo",  # used module file name
        MODEL_CFG="yolov4l-mish.yaml",
        NUM_CLASSES=NUM_CLASSES,
        IOU_THR_TRAIN=0.20,  # iou training threshold
        ANCHOR_THR=4.0,  # anchor-multiple threshold
        CONF_THR_TEST=0.001,
        IOU_THR_TEST=0.65,  # iou test threshold
        USE_MERGE_NMS=False,
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
    MASK_ON=False,
    KEYPOINT_ON=False,
    LOAD_PROPOSALS=False,
)

TEST = dict(EVAL_PERIOD=0, VIS=False, PRECISE_BN=dict(ENABLED=False, NUM_ITER=200))
