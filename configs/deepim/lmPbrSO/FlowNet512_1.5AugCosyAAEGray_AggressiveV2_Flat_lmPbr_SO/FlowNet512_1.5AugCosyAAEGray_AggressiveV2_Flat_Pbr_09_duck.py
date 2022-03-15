_base_ = "./FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_01_ape.py"
OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/duck"
DATASETS = dict(TRAIN=("lm_pbr_duck_train",), TEST=("lm_real_duck_test",))

# bbnc6
# objects  duck    Avg(1)
# ad_2     4.23    4.23
# ad_5     26.01   26.01
# ad_10    61.88   61.88
# rete_2   54.37   54.37
# rete_5   97.28   97.28
# rete_10  100.00  100.00
# re_2     59.15   59.15
# re_5     97.28   97.28
# re_10    100.00  100.00
# te_2     89.67   89.67
# te_5     100.00  100.00
# te_10    100.00  100.00
# proj_2   85.63   85.63
# proj_5   98.22   98.22
# proj_10  99.91   99.91
# re       2.02    2.02
# te       0.01    0.01
