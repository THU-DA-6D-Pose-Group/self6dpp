_base_ = "./FlowNet512_1.5AugCosyAAEGray_AdamW_ClipGrad_AvgMaxMin_Pbr_01_ape.py"
OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AdamW_ClipGrad_AvgMaxMin_lmPbr_SO/cat"
DATASETS = dict(TRAIN=("lm_pbr_cat_train",), TEST=("lm_real_cat_test",))

# bbnc10
# objects  cat     Avg(1)
# ad_2     21.06   21.06
# ad_5     62.67   62.67
# ad_10    92.61   92.61
# rete_2   78.04   78.04
# rete_5   99.20   99.20
# rete_10  100.00  100.00
# re_2     78.34   78.34
# re_5     99.20   99.20
# re_10    100.00  100.00
# te_2     99.10   99.10
# te_5     100.00  100.00
# te_10    100.00  100.00
# proj_2   85.83   85.83
# proj_5   99.10   99.10
# proj_10  100.00  100.00
# re       1.49    1.49
# te       0.01    0.01
