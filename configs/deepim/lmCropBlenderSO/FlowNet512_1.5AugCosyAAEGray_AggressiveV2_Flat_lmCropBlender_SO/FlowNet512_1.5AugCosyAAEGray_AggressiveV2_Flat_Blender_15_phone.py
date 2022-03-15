_base_ = "./FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Blender_01_ape.py"
OUTPUT_DIR = "output/deepim/lmCropBlenderSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmCropBlender_SO/phone"
DATASETS = dict(TRAIN=("lm_blender_phone_train",), TEST=("lm_crop_phone_test",))

# bbnc6
# iter0
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
#
# iter4
# objects  phone   Avg(1)
# ad_2     5.22    5.22
# ad_5     66.67   66.67
# ad_10    97.19   97.19
# rete_2   36.14   36.14
# rete_5   90.76   90.76
# rete_10  100.00  100.00
# re_2     37.35   37.35
# re_5     90.76   90.76
# re_10    100.00  100.00
# te_2     95.58   95.58
# te_5     99.20   99.20
# te_10    100.00  100.00
# proj_2   16.06   16.06
# proj_5   95.58   95.58
# proj_10  100.00  100.00
# re       2.76    2.76
# te       0.01    0.01
