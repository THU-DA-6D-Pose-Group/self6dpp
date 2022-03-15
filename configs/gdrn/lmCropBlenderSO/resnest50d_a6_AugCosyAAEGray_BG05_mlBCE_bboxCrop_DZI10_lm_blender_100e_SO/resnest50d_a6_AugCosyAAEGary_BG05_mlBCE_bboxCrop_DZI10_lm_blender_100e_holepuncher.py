_base_ = ["./resnest50d_a6_AugCosyAAEGary_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e_ape.py"]

OUTPUT_DIR = (
    "output/gdrn/lm_crop_blender/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e/holepuncher"
)

DATASETS = dict(
    TRAIN=("lm_blender_holepuncher_train",),
    TEST=("lm_crop_holepuncher_test",),
)

# objects  holepuncher  Avg(1)
# ad_2     0.00         0.00
# ad_5     0.00         0.00
# ad_10    0.00         0.00
# rete_2   0.00         0.00
# rete_5   23.39        23.39
# rete_10  52.82        52.82
# re_2     4.84         4.84
# re_5     40.73        40.73
# re_10    78.23        78.23
# te_2     1.21         1.21
# te_5     33.87        33.87
# te_10    58.47        58.47
# proj_2   0.00         0.00
# proj_5   4.03         4.03
# proj_10  33.87        33.87
# re       7.44         7.44
# te       0.09         0.09
