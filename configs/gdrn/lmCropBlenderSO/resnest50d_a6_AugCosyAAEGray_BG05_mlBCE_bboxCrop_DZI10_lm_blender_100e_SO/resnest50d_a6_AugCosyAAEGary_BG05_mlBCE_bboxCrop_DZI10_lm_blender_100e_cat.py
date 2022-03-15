_base_ = ["./resnest50d_a6_AugCosyAAEGary_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e_ape.py"]

OUTPUT_DIR = "output/gdrn/lm_crop_blender/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e/cat"

DATASETS = dict(
    TRAIN=("lm_blender_cat_train",),
    TEST=("lm_crop_cat_test",),
)

# objects  cat    Avg(1)
# ad_2     2.97   2.97
# ad_5     22.46  22.46
# ad_10    35.17  35.17
# rete_2   22.46  22.46
# rete_5   50.42  50.42
# rete_10  66.53  66.53
# re_2     24.58  24.58
# re_5     66.53  66.53
# re_10    84.32  84.32
# te_2     38.56  38.56
# te_5     53.39  53.39
# te_10    69.49  69.49
# proj_2   16.95  16.95
# proj_5   73.73  73.73
# proj_10  91.95  91.95
# re       9.27   9.27
# te       0.07   0.07
