_base_ = ["./resnest50d_a6_AugCosyAAEGary_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e_ape.py"]

OUTPUT_DIR = "output/gdrn/lm_crop_blender/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e/phone"

DATASETS = dict(
    TRAIN=("lm_blender_phone_train",),
    TEST=("lm_crop_phone_test",),
)

# objects  phone  Avg(1)
# ad_2     2.01   2.01
# ad_5     38.55  38.55
# ad_10    74.30  74.30
# rete_2   18.07  18.07
# rete_5   76.71  76.71
# rete_10  97.59  97.59
# re_2     22.09  22.09
# re_5     79.12  79.12
# re_10    97.99  97.99
# te_2     70.68  70.68
# te_5     95.58  95.58
# te_10    99.20  99.20
# proj_2   8.03   8.03
# proj_5   89.56  89.56
# proj_10  99.20  99.20
# re       3.73   3.73
# te       0.02   0.02
