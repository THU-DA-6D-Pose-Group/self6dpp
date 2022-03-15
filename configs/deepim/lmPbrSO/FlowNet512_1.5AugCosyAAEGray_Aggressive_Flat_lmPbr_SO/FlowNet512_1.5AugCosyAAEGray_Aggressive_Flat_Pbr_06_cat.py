_base_ = "./FlowNet512_1.5AugCosyAAEGray_Aggressive_Flat_Pbr_01_ape.py"
OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_Aggressive_Flat_lmPbr_SO/cat"
DATASETS = dict(TRAIN=("lm_pbr_cat_train",), TEST=("lm_real_cat_test",))

# rl1
# objects  cat     Avg(1)
# ad_2     20.76   20.76
# ad_5     65.17   65.17
# ad_10    93.11   93.11
# rete_2   74.85   74.85
# rete_5   98.80   98.80
# rete_10  100.00  100.00
# re_2     75.95   75.95
# re_5     98.80   98.80
# re_10    100.00  100.00
# te_2     98.30   98.30
# te_5     100.00  100.00
# te_10    100.00  100.00
# proj_2   84.53   84.53
# proj_5   99.00   99.00
# proj_10  100.00  100.00
# re       1.56    1.56
# te       0.01    0.01
