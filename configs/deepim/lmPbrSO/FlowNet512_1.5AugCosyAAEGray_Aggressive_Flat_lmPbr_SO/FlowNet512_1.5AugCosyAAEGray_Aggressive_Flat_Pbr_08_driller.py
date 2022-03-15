_base_ = "./FlowNet512_1.5AugCosyAAEGray_Aggressive_Flat_Pbr_01_ape.py"
OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_Aggressive_Flat_lmPbr_SO/driller"
DATASETS = dict(TRAIN=("lm_pbr_driller_train",), TEST=("lm_real_driller_test",))

# rl1
# objects  driller  Avg(1)
# ad_2     48.27    48.27
# ad_5     92.67    92.67
# ad_10    100.00   100.00
# rete_2   87.61    87.61
# rete_5   99.90    99.90
# rete_10  100.00   100.00
# re_2     87.91    87.91
# re_5     99.90    99.90
# re_10    100.00   100.00
# te_2     99.41    99.41
# te_5     100.00   100.00
# te_10    100.00   100.00
# proj_2   84.84    84.84
# proj_5   98.71    98.71
# proj_10  100.00   100.00
# re       1.23     1.23
# te       0.01     0.01
