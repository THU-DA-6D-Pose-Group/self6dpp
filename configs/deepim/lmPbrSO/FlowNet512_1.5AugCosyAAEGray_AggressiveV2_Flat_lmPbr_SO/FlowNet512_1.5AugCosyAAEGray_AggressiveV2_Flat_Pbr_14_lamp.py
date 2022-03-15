_base_ = "./FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_01_ape.py"
OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/lamp"
DATASETS = dict(TRAIN=("lm_pbr_lamp_train",), TEST=("lm_real_lamp_test",))

# bbnc6
# objects  lamp   Avg(1)
# ad_2     0.77   0.77
# ad_5     37.43  37.43
# ad_10    99.14  99.14
# rete_2   42.03  42.03
# rete_5   99.81  99.81
# rete_10  99.90  99.90
# re_2     48.66  48.66
# re_5     99.81  99.81
# re_10    99.90  99.90
# te_2     85.41  85.41
# te_5     99.90  99.90
# te_10    99.90  99.90
# proj_2   19.96  19.96
# proj_5   96.55  96.55
# proj_10  99.90  99.90
# re       2.13   2.13
# te       0.02   0.02
