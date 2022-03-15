_base_ = "./FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_01_ape.py"
OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/can"
DATASETS = dict(TRAIN=("lm_pbr_can_train",), TEST=("lm_real_can_test",))

# bbnc5
# objects  can     Avg(1)
# ad_2     15.94   15.94
# ad_5     89.47   89.47
# ad_10    99.80   99.80
# rete_2   97.54   97.54
# rete_5   100.00  100.00
# rete_10  100.00  100.00
# re_2     97.74   97.74
# re_5     100.00  100.00
# re_10    100.00  100.00
# te_2     99.80   99.80
# te_5     100.00  100.00
# te_10    100.00  100.00
# proj_2   44.88   44.88
# proj_5   98.62   98.62
# proj_10  100.00  100.00
# re       0.90    0.90
# te       0.01    0.01
