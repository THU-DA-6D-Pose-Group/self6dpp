_base_ = ["./resnest50d_a6_AugCosyAAEGary_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e_ape.py"]

OUTPUT_DIR = "output/gdrn/lm_crop_blender/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e/duck"

DATASETS = dict(
    TRAIN=("lm_blender_duck_train",),
    TEST=("lm_crop_duck_test",),
)

# objects  duck   Avg(1)
# ad_2     0.00   0.00
# ad_5     0.00   0.00
# ad_10    0.40   0.40
# rete_2   0.80   0.80
# rete_5   11.16  11.16
# rete_10  25.90  25.90
# re_2     3.59   3.59
# re_5     17.13  17.13
# re_10    32.67  32.67
# te_2     0.80   0.80
# te_5     15.54  15.54
# te_10    28.69  28.69
# proj_2   0.80   0.80
# proj_5   24.30  24.30
# proj_10  43.82  43.82
# re       39.39  39.39
# te       0.24   0.24
