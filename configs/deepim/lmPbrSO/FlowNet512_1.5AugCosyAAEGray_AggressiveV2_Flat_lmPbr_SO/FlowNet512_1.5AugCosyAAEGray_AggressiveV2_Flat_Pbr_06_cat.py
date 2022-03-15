_base_ = "./FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_01_ape.py"
OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/cat"
DATASETS = dict(TRAIN=("lm_pbr_cat_train",), TEST=("lm_real_cat_test",))

# bbnc5
# objects  cat     Avg(1)
# ad_2     19.26   19.26
# ad_5     62.48   62.48
# ad_10    91.52   91.52
# rete_2   78.24   78.24
# rete_5   99.60   99.60
# rete_10  100.00  100.00
# re_2     79.74   79.74
# re_5     99.60   99.60
# re_10    100.00  100.00
# te_2     97.70   97.70
# te_5     100.00  100.00
# te_10    100.00  100.00
# proj_2   83.63   83.63
# proj_5   99.10   99.10
# proj_10  100.00  100.00
# re       1.45    1.45
# te       0.01    0.01
