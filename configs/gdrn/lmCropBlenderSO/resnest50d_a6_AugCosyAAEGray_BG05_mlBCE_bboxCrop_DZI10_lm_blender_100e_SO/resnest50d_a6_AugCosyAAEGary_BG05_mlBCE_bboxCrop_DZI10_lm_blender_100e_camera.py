_base_ = ["./resnest50d_a6_AugCosyAAEGary_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e_ape.py"]

OUTPUT_DIR = "output/gdrn/lm_crop_blender/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e/camera"

DATASETS = dict(
    TRAIN=("lm_blender_camera_train",),
    TEST=("lm_crop_camera_test",),
)

# objects  camera  Avg(1)
# ad_2     4.56    4.56
# ad_5     23.24   23.24
# ad_10    43.57   43.57
# rete_2   14.94   14.94
# rete_5   63.07   63.07
# rete_10  90.46   90.46
# re_2     21.16   21.16
# re_5     75.52   75.52
# re_10    96.27   96.27
# te_2     47.30   47.30
# te_5     71.78   71.78
# te_10    93.78   93.78
# proj_2   26.97   26.97
# proj_5   84.23   84.23
# proj_10  99.17   99.17
# re       3.98    3.98
# te       0.03    0.03
