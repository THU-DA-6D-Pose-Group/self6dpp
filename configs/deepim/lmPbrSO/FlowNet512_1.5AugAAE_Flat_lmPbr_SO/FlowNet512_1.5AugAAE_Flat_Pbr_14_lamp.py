_base_ = ["./FlowNet512_1.5AugAAE_Flat_Pbr_01_ape.py"]


OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugAAE_Flat_lmPbr_SO/lamp"

DATASETS = dict(TRAIN=("lm_pbr_lamp_train",), TEST=("lm_real_lamp_test",))

# gu
# objects  lamp   Avg(1)
# ad_2     0.10   0.10
# ad_5     40.79  40.79
# ad_10    99.33  99.33
# rete_2   44.91  44.91
# rete_5   99.81  99.81
# rete_10  99.90  99.90
# re_2     50.00  50.00
# re_5     99.81  99.81
# re_10    99.90  99.90
# te_2     89.73  89.73
# te_5     99.90  99.90
# te_10    99.90  99.90
# proj_2   18.81  18.81
# proj_5   96.26  96.26
# proj_10  99.90  99.90
# re       2.06   2.06
# te       0.02   0.02
