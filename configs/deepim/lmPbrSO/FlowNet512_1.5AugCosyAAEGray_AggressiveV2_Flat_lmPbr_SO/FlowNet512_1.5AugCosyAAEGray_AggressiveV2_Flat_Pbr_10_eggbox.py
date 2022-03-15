_base_ = "./FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_01_ape.py"
OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/eggbox"
DATASETS = dict(TRAIN=("lm_pbr_eggbox_train",), TEST=("lm_real_eggbox_test",))

# bbnc6
# objects  eggbox  Avg(1)
# ad_2     11.46   11.46
# ad_5     52.02   52.02
# ad_10    93.52   93.52
# rete_2   47.98   47.98
# rete_5   95.12   95.12
# rete_10  99.81   99.81
# re_2     65.26   65.26
# re_5     95.40   95.40
# re_10    99.81   99.81
# te_2     60.85   60.85
# te_5     98.78   98.78
# te_10    100.00  100.00
# proj_2   60.09   60.09
# proj_5   98.50   98.50
# proj_10  99.91   99.91
# re       2.01    2.01
# te       0.02    0.02
