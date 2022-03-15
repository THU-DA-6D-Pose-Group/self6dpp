_base_ = ["./FlowNet512_1.5AugAAE_Flat_Pbr_01_ape.py"]


OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugAAE_Flat_lmPbr_SO/glue"

DATASETS = dict(TRAIN=("lm_pbr_glue_train",), TEST=("lm_real_glue_test",))

# rl1
# objects  glue   Avg(1)
# ad_2     5.21   5.21
# ad_5     46.24  46.24
# ad_10    80.79  80.79
# rete_2   1.45   1.45
# rete_5   19.40  19.40
# rete_10  66.70  66.70
# re_2     1.83   1.83
# re_5     20.17  20.17
# re_10    66.70  66.70
# te_2     54.73  54.73
# te_5     92.37  92.37
# te_10    99.90  99.90
# proj_2   1.35   1.35
# proj_5   66.02  66.02
# proj_10  97.01  97.01
# re       9.06   9.06
# te       0.02   0.02
