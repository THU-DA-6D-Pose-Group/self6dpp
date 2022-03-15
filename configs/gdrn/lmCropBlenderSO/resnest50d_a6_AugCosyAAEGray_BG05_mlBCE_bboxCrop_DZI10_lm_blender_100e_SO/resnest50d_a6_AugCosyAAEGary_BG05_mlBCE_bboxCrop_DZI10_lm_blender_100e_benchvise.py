_base_ = ["./resnest50d_a6_AugCosyAAEGary_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e_ape.py"]

OUTPUT_DIR = (
    "output/gdrn/lm_crop_blender/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e/benchvise"
)

DATASETS = dict(
    TRAIN=("lm_blender_benchvise_train",),
    TEST=("lm_crop_benchvise_test",),
)

# objects  benchvise  Avg(1)
# ad_2     11.52      11.52
# ad_5     55.97      55.97
# ad_10    90.53      90.53
# rete_2   36.21      36.21
# rete_5   98.35      98.35
# rete_10  100.00     100.00
# re_2     46.50      46.50
# re_5     98.35      98.35
# re_10    100.00     100.00
# te_2     83.54      83.54
# te_5     100.00     100.00
# te_10    100.00     100.00
# proj_2   57.61      57.61
# proj_5   99.18      99.18
# proj_10  100.00     100.00
# re       2.24       2.24
# te       0.01       0.01
