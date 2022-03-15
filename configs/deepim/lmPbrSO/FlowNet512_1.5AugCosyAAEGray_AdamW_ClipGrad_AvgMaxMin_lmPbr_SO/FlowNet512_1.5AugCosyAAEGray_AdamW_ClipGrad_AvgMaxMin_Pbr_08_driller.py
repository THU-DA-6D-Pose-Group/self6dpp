_base_ = "./FlowNet512_1.5AugCosyAAEGray_AdamW_ClipGrad_AvgMaxMin_Pbr_01_ape.py"
OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AdamW_ClipGrad_AvgMaxMin_lmPbr_SO/driller"
DATASETS = dict(TRAIN=("lm_pbr_driller_train",), TEST=("lm_real_driller_test",))

# bbnc10
# objects  driller  Avg(1)
# ad_2     51.14    51.14
# ad_5     94.15    94.15
# ad_10    100.00   100.00
# rete_2   91.58    91.58
# rete_5   99.90    99.90
# rete_10  100.00   100.00
# re_2     91.77    91.77
# re_5     99.90    99.90
# re_10    100.00   100.00
# te_2     99.60    99.60
# te_5     100.00   100.00
# te_10    100.00   100.00
# proj_2   86.22    86.22
# proj_5   98.81    98.81
# proj_10  100.00   100.00
# re       1.12     1.12
# te       0.01     0.01
