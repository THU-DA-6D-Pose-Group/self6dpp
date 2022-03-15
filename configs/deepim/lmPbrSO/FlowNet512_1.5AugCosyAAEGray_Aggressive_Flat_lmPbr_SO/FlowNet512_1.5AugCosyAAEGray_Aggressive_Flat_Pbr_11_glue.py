_base_ = "./FlowNet512_1.5AugCosyAAEGray_Aggressive_Flat_Pbr_01_ape.py"
OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_Aggressive_Flat_lmPbr_SO/glue"
DATASETS = dict(TRAIN=("lm_pbr_glue_train",), TEST=("lm_real_glue_test",))

# rl1
# objects  glue   Avg(1)
# ad_2     20.46  20.46
# ad_5     62.64  62.64
# ad_10    89.77  89.77
# rete_2   6.27   6.27
# rete_5   73.36  73.36
# rete_10  96.53  96.53
# re_2     8.69   8.69
# re_5     75.58  75.58
# re_10    96.81  96.81
# te_2     62.64  62.64
# te_5     94.88  94.88
# te_10    99.61  99.61
# proj_2   20.95  20.95
# proj_5   94.11  94.11
# proj_10  99.90  99.90
# re       4.21   4.21
# te       0.02   0.02
