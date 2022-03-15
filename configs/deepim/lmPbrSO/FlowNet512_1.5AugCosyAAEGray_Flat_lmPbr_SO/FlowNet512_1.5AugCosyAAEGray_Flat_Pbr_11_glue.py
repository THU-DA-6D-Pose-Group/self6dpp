_base_ = ["./FlowNet512_1.5AugCosyAAEGray_Flat_Pbr_01_ape.py"]


OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_Flat_lmPbr_SO/glue"

DATASETS = dict(TRAIN=("lm_pbr_glue_train",), TEST=("lm_real_glue_test",))

# bbnc9
# objects  glue   Avg(1)
# ad_2     20.95  20.95
# ad_5     59.46  59.46
# ad_10    89.58  89.58
# rete_2   5.79   5.79
# rete_5   70.17  70.17
# rete_10  97.30  97.30
# re_2     8.11   8.11
# re_5     73.46  73.46
# re_10    97.39  97.39
# te_2     62.64  62.64
# te_5     94.31  94.31
# te_10    99.81  99.81
# proj_2   19.11  19.11
# proj_5   92.28  92.28
# proj_10  99.61  99.61
# re       4.32   4.32
# te       0.02   0.02

# init by mlBCE
# objects  glue   Avg(1)
# ad_2     21.43  21.43
# ad_5     63.90  63.90
# ad_10    90.73  90.73
# rete_2   7.24   7.24
# rete_5   74.03  74.03
# rete_10  98.17  98.17
# re_2     9.27   9.27
# re_5     76.45  76.45
# re_10    98.26  98.26
# te_2     64.58  64.58
# te_5     95.75  95.75
# te_10    99.81  99.81
# proj_2   21.04  21.04
# proj_5   93.73  93.73
# proj_10  99.90  99.90
# re       3.97   3.97
# te       0.02   0.02
