_base_ = "./FlowNet512_1.5AugCosyAAEGray_AdamW_ClipGrad_AvgMaxMin_Pbr_01_ape.py"
OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AdamW_ClipGrad_AvgMaxMin_lmPbr_SO/can"
DATASETS = dict(TRAIN=("lm_pbr_can_train",), TEST=("lm_real_can_test",))

# bbnc10
# objects  can     Avg(1)
# ad_2     17.72   17.72
# ad_5     87.89   87.89
# ad_10    99.61   99.61
# rete_2   97.34   97.34
# rete_5   100.00  100.00
# rete_10  100.00  100.00
# re_2     97.74   97.74
# re_5     100.00  100.00
# re_10    100.00  100.00
# te_2     99.61   99.61
# te_5     100.00  100.00
# te_10    100.00  100.00
# proj_2   45.96   45.96
# proj_5   98.62   98.62
# proj_10  100.00  100.00
# re       0.88    0.88
# te       0.01    0.01
