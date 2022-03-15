_base_ = "./FlowNet512_1.5AugCosyAAEGray_AdamW_ClipGrad_AvgMaxMin_Pbr_01_ape.py"
OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AdamW_ClipGrad_AvgMaxMin_lmPbr_SO/benchvise"
DATASETS = dict(TRAIN=("lm_pbr_benchvise_train",), TEST=("lm_real_benchvise_test",))

# bbnc7
# objects  benchvise  Avg(1)
# ad_2     11.06      11.06
# ad_5     48.11      48.11
# ad_10    94.18      94.18
# rete_2   45.88      45.88
# rete_5   99.32      99.32
# rete_10  100.00     100.00
# re_2     58.20      58.20
# re_5     99.32      99.32
# re_10    100.00     100.00
# te_2     81.86      81.86
# te_5     100.00     100.00
# te_10    100.00     100.00
# proj_2   71.39      71.39
# proj_5   99.22      99.22
# proj_10  100.00     100.00
# re       1.96       1.96
# te       0.01       0.01
