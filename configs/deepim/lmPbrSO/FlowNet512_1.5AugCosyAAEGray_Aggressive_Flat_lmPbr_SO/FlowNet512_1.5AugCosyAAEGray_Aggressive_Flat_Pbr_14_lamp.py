_base_ = "./FlowNet512_1.5AugCosyAAEGray_Aggressive_Flat_Pbr_01_ape.py"
OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_Aggressive_Flat_lmPbr_SO/lamp"
DATASETS = dict(TRAIN=("lm_pbr_lamp_train",), TEST=("lm_real_lamp_test",))

# gu
# objects  lamp   Avg(1)
# ad_2     0.96   0.96
# ad_5     43.95  43.95
# ad_10    99.42  99.42
# rete_2   41.36  41.36
# rete_5   99.81  99.81
# rete_10  99.90  99.90
# re_2     46.55  46.55
# re_5     99.81  99.81
# re_10    99.90  99.90
# te_2     88.68  88.68
# te_5     99.90  99.90
# te_10    99.90  99.90
# proj_2   19.87  19.87
# proj_5   96.35  96.35
# proj_10  99.90  99.90
# re       2.15   2.15
# te       0.02   0.02
