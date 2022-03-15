_base_ = ["./resnest50d_a6_AugCosyAAEGary_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e_ape.py"]

OUTPUT_DIR = (
    "output/gdrn/lm_crop_blender/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e/driller"
)

DATASETS = dict(
    TRAIN=("lm_blender_driller_train",),
    TEST=("lm_crop_driller_test",),
)

# objects  driller  Avg(1)
# ad_2     15.97    15.97
# ad_5     55.46    55.46
# ad_10    81.09    81.09
# rete_2   34.87    34.87
# rete_5   83.61    83.61
# rete_10  96.22    96.22
# re_2     40.34    40.34
# re_5     84.87    84.87
# re_10    96.22    96.22
# te_2     73.11    73.11
# te_5     94.12    94.12
# te_10    100.00   100.00
# proj_2   44.54    44.54
# proj_5   88.66    88.66
# proj_10  97.06    97.06
# re       4.69     4.69
# te       0.02     0.02
