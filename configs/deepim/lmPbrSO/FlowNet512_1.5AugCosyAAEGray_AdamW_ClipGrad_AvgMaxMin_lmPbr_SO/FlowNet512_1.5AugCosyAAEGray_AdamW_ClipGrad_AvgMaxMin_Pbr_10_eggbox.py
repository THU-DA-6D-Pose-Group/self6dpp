_base_ = "./FlowNet512_1.5AugCosyAAEGray_AdamW_ClipGrad_AvgMaxMin_Pbr_01_ape.py"
OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AdamW_ClipGrad_AvgMaxMin_lmPbr_SO/eggbox"
DATASETS = dict(TRAIN=("lm_pbr_eggbox_train",), TEST=("lm_real_eggbox_test",))

# bbnc4
# objects  eggbox  Avg(1)
# ad_2     7.89    7.89
# ad_5     38.40   38.40
# ad_10    89.20   89.20
# rete_2   3.57    3.57
# rete_5   37.00   37.00
# rete_10  79.62   79.62
# re_2     6.48    6.48
# re_5     37.75   37.75
# re_10    79.62   79.62
# te_2     44.79   44.79
# te_5     96.71   96.71
# te_10    100.00  100.00
# proj_2   32.58   32.58
# proj_5   89.95   89.95
# proj_10  99.34   99.34
# re       6.91    6.91
# te       0.02    0.02
