_base_ = ["./resnest50d_a6_AugCosyAAEGary_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e_ape.py"]

OUTPUT_DIR = "output/gdrn/lm_crop_blender/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e/lamp"

DATASETS = dict(
    TRAIN=("lm_blender_lamp_train",),
    TEST=("lm_crop_lamp_test",),
)

# objects  lamp   Avg(1)
# ad_2     6.10   6.10
# ad_5     36.99  36.99
# ad_10    70.73  70.73
# rete_2   10.98  10.98
# rete_5   71.95  71.95
# rete_10  91.87  91.87
# re_2     15.45  15.45
# re_5     72.36  72.36
# re_10    93.09  93.09
# te_2     54.47  54.47
# te_5     88.62  88.62
# te_10    95.12  95.12
# proj_2   6.91   6.91
# proj_5   78.46  78.46
# proj_10  94.31  94.31
# re       5.42   5.42
# te       0.03   0.03
