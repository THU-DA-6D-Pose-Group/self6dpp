_base_ = "./FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Blender_01_ape.py"
OUTPUT_DIR = "output/deepim/lmCropBlenderSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmCropBlender_SO/lamp"
DATASETS = dict(TRAIN=("lm_blender_lamp_train",), TEST=("lm_crop_lamp_test",))

# bbnc6
# iter0
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
#
# iter4
# objects  lamp   Avg(1)
# ad_2     1.63   1.63
# ad_5     30.89  30.89
# ad_10    90.24  90.24
# rete_2   24.80  24.80
# rete_5   95.93  95.93
# rete_10  97.56  97.56
# re_2     36.18  36.18
# re_5     95.93  95.93
# re_10    97.56  97.56
# te_2     63.01  63.01
# te_5     97.97  97.97
# te_10    98.37  98.37
# proj_2   11.38  11.38
# proj_5   91.87  91.87
# proj_10  97.56  97.56
# re       3.95   3.95
# te       0.02   0.02
