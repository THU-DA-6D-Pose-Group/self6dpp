_base_ = "./FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_01_ape.py"
OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/driller"
DATASETS = dict(TRAIN=("lm_pbr_driller_train",), TEST=("lm_real_driller_test",))

# bbnc7
# objects  driller  Avg(1)
# ad_2     51.24    51.24
# ad_5     92.57    92.57
# ad_10    100.00   100.00
# rete_2   92.86    92.86
# rete_5   99.90    99.90
# rete_10  100.00   100.00
# re_2     93.06    93.06
# re_5     99.90    99.90
# re_10    100.00   100.00
# te_2     99.70    99.70
# te_5     100.00   100.00
# te_10    100.00   100.00
# proj_2   86.03    86.03
# proj_5   98.91    98.91
# proj_10  100.00   100.00
# re       1.12     1.12
# te       0.01     0.01
