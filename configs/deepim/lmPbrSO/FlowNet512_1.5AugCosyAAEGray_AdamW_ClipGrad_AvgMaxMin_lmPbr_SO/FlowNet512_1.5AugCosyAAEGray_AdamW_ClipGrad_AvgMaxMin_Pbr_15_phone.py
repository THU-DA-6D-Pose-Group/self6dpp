_base_ = "./FlowNet512_1.5AugCosyAAEGray_AdamW_ClipGrad_AvgMaxMin_Pbr_01_ape.py"
OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AdamW_ClipGrad_AvgMaxMin_lmPbr_SO/phone"
DATASETS = dict(TRAIN=("lm_pbr_phone_train",), TEST=("lm_real_phone_test",))

# rl3
# objects  phone   Avg(1)
# ad_2     5.85    5.85
# ad_5     63.27   63.27
# ad_10    94.24   94.24
# rete_2   34.28   34.28
# rete_5   96.13   96.13
# rete_10  100.00  100.00
# re_2     37.39   37.39
# re_5     96.13   96.13
# re_10    100.00  100.00
# te_2     91.88   91.88
# te_5     100.00  100.00
# te_10    100.00  100.00
# proj_2   26.53   26.53
# proj_5   97.83   97.83
# proj_10  100.00  100.00
# re       2.56    2.56
# te       0.01    0.01
