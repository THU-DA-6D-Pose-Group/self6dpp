_base_ = "./FlowNet512_1.5AugCosyAAEGray_Aggressive_Flat_Pbr_01_ape.py"
OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_Aggressive_Flat_lmPbr_SO/can"
DATASETS = dict(TRAIN=("lm_pbr_can_train",), TEST=("lm_real_can_test",))

# rl1
# objects  can     Avg(1)
# ad_2     17.42   17.42
# ad_5     86.12   86.12
# ad_10    99.90   99.90
# rete_2   97.15   97.15
# rete_5   100.00  100.00
# rete_10  100.00  100.00
# re_2     97.24   97.24
# re_5     100.00  100.00
# re_10    100.00  100.00
# te_2     99.90   99.90
# te_5     100.00  100.00
# te_10    100.00  100.00
# proj_2   47.15   47.15
# proj_5   98.52   98.52
# proj_10  100.00  100.00
# re       0.94    0.94
# te       0.01    0.01
