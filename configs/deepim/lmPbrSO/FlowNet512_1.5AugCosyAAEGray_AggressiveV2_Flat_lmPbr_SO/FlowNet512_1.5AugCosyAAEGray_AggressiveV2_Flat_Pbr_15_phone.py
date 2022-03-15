_base_ = "./FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_01_ape.py"
OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/phone"
DATASETS = dict(TRAIN=("lm_pbr_phone_train",), TEST=("lm_real_phone_test",))

# bbnc7
# objects  phone   Avg(1)
# ad_2     7.46    7.46
# ad_5     69.59   69.59
# ad_10    94.81   94.81
# rete_2   37.39   37.39
# rete_5   97.26   97.26
# rete_10  100.00  100.00
# re_2     40.04   40.04
# re_5     97.26   97.26
# re_10    100.00  100.00
# te_2     93.20   93.20
# te_5     100.00  100.00
# te_10    100.00  100.00
# proj_2   23.23   23.23
# proj_5   97.92   97.92
# proj_10  100.00  100.00
# re       2.49    2.49
# te       0.01    0.01
