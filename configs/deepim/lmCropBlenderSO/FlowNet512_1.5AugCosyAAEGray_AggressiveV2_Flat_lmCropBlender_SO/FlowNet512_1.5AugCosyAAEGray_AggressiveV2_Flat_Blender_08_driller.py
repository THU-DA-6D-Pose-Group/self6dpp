_base_ = "./FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Blender_01_ape.py"
OUTPUT_DIR = "output/deepim/lmCropBlenderSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmCropBlender_SO/driller"
DATASETS = dict(TRAIN=("lm_blender_driller_train",), TEST=("lm_crop_driller_test",))

# iter0
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

# iter4
# objects  driller  Avg(1)
# ad_2     39.08    39.08
# ad_5     85.71    85.71
# ad_10    98.74    98.74
# rete_2   80.25    80.25
# rete_5   97.06    97.06
# rete_10  99.16    99.16
# re_2     81.09    81.09
# re_5     97.06    97.06
# re_10    99.16    99.16
# te_2     96.64    96.64
# te_5     99.16    99.16
# te_10    99.58    99.58
# proj_2   80.67    80.67
# proj_5   97.48    97.48
# proj_10  99.16    99.16
# re       2.90     2.90
# te       0.01     0.01
