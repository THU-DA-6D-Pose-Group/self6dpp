_base_ = ["./resnest50d_a6_AugCosyAAEGary_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e_ape.py"]

OUTPUT_DIR = "output/gdrn/lm_crop_blender/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e/iron"

DATASETS = dict(
    TRAIN=("lm_blender_iron_train",),
    TEST=("lm_crop_iron_test",),
)

# objects  iron    Avg(1)
# ad_2     5.19    5.19
# ad_5     67.97   67.97
# ad_10    94.37   94.37
# rete_2   32.90   32.90
# rete_5   87.45   87.45
# rete_10  95.67   95.67
# re_2     35.06   35.06
# re_5     87.88   87.88
# re_10    95.67   95.67
# te_2     87.45   87.45
# te_5     99.57   99.57
# te_10    100.00  100.00
# proj_2   6.06    6.06
# proj_5   61.90   61.90
# proj_10  99.13   99.13
# re       3.25    3.25
# te       0.01    0.01
