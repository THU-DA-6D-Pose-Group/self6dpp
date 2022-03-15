_base_ = ["./FlowNet512_1.5AugCosyAAEGray_Flat_Pbr_01_ape.py"]


OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_Flat_lmPbr_SO/cat"

DATASETS = dict(TRAIN=("lm_pbr_cat_train",), TEST=("lm_real_cat_test",))

# bbnc10
# objects  cat     Avg(1)
# ad_2     19.86   19.86
# ad_5     62.67   62.67
# ad_10    91.92   91.92
# rete_2   74.45   74.45
# rete_5   98.80   98.80
# rete_10  100.00  100.00
# re_2     75.65   75.65
# re_5     98.80   98.80
# re_10    100.00  100.00
# te_2     98.10   98.10
# te_5     100.00  100.00
# te_10    100.00  100.00
# proj_2   85.43   85.43
# proj_5   99.10   99.10
# proj_10  100.00  100.00
# re       1.57    1.57
# te       0.01    0.01

# init by mlBCE
# objects  cat     Avg(1)
# ad_2     20.36   20.36
# ad_5     62.87   62.87
# ad_10    92.02   92.02
# rete_2   74.35   74.35
# rete_5   98.90   98.90
# rete_10  100.00  100.00
# re_2     75.65   75.65
# re_5     98.90   98.90
# re_10    100.00  100.00
# te_2     97.90   97.90
# te_5     100.00  100.00
# te_10    100.00  100.00
# proj_2   85.83   85.83
# proj_5   99.10   99.10
# proj_10  100.00  100.00
# re       1.57    1.57
# te       0.01    0.01
