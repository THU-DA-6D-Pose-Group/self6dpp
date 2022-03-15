_base_ = ["./resnest50d_a6_AugCosyAAEGary_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e_ape.py"]

OUTPUT_DIR = "output/gdrn/lm_crop_blender/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e/can"

DATASETS = dict(
    TRAIN=("lm_blender_can_train",),
    TEST=("lm_crop_can_test",),
)

# objects  can     Avg(1)
# ad_2     17.92   17.92
# ad_5     79.17   79.17
# ad_10    98.33   98.33
# rete_2   57.92   57.92
# rete_5   98.33   98.33
# rete_10  100.00  100.00
# re_2     57.92   57.92
# re_5     98.33   98.33
# re_10    100.00  100.00
# te_2     97.92   97.92
# te_5     100.00  100.00
# te_10    100.00  100.00
# proj_2   20.42   20.42
# proj_5   96.67   96.67
# proj_10  100.00  100.00
# re       1.96    1.96
# te       0.01    0.01
