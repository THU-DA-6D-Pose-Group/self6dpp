_base_ = "./FlowNet512_1.5AugCosyAAEGray_AdamW_ClipGrad_AvgMaxMin_Pbr_01_ape.py"
OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AdamW_ClipGrad_AvgMaxMin_lmPbr_SO/camera"
DATASETS = dict(TRAIN=("lm_pbr_camera_train",), TEST=("lm_real_camera_test",))

# bbnc7
# objects  camera  Avg(1)
# ad_2     28.63   28.63
# ad_5     84.12   84.12
# ad_10    98.73   98.73
# rete_2   42.84   42.84
# rete_5   99.31   99.31
# rete_10  100.00  100.00
# re_2     42.94   42.94
# re_5     99.31   99.31
# re_10    100.00  100.00
# te_2     99.71   99.71
# te_5     100.00  100.00
# te_10    100.00  100.00
# proj_2   84.61   84.61
# proj_5   99.31   99.31
# proj_10  100.00  100.00
# re       2.26    2.26
# te       0.01    0.01
