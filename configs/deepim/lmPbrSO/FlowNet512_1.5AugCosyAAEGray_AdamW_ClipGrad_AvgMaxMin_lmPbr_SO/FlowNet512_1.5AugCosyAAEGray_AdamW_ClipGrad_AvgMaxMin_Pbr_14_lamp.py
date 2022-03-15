_base_ = "./FlowNet512_1.5AugCosyAAEGray_AdamW_ClipGrad_AvgMaxMin_Pbr_01_ape.py"
OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AdamW_ClipGrad_AvgMaxMin_lmPbr_SO/lamp"
DATASETS = dict(TRAIN=("lm_pbr_lamp_train",), TEST=("lm_real_lamp_test",))

# rl3
# objects  lamp   Avg(1)
# ad_2     0.96   0.96
# ad_5     38.58  38.58
# ad_10    99.04  99.04
# rete_2   49.04  49.04
# rete_5   99.81  99.81
# rete_10  99.90  99.90
# re_2     54.61  54.61
# re_5     99.81  99.81
# re_10    99.90  99.90
# te_2     88.68  88.68
# te_5     99.90  99.90
# te_10    99.90  99.90
# proj_2   20.06  20.06
# proj_5   96.64  96.64
# proj_10  99.90  99.90
# re       2.04   2.04
# te       0.02   0.02
