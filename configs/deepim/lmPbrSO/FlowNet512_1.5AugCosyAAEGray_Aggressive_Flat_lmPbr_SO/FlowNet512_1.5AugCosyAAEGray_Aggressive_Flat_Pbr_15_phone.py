_base_ = "./FlowNet512_1.5AugCosyAAEGray_Aggressive_Flat_Pbr_01_ape.py"
OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_Aggressive_Flat_lmPbr_SO/phone"
DATASETS = dict(TRAIN=("lm_pbr_phone_train",), TEST=("lm_real_phone_test",))

# gu
# objects  phone   Avg(1)
# ad_2     6.42    6.42
# ad_5     64.49   64.49
# ad_10    93.11   93.11
# rete_2   34.28   34.28
# rete_5   95.00   95.00
# rete_10  100.00  100.00
# re_2     36.92   36.92
# re_5     95.00   95.00
# re_10    100.00  100.00
# te_2     91.88   91.88
# te_5     100.00  100.00
# te_10    100.00  100.00
# proj_2   24.93   24.93
# proj_5   97.73   97.73
# proj_10  100.00  100.00
# re       2.62    2.62
# te       0.01    0.01
