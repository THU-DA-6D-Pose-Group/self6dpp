_base_ = "./FlowNet512_1.5AugCosyAAEGray_Aggressive_Flat_Pbr_01_ape.py"
OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_Aggressive_Flat_lmPbr_SO/duck"
DATASETS = dict(TRAIN=("lm_pbr_duck_train",), TEST=("lm_real_duck_test",))

# rl1
# objects  duck    Avg(1)
# ad_2     4.32    4.32
# ad_5     29.67   29.67
# ad_10    63.57   63.57
# rete_2   54.84   54.84
# rete_5   96.90   96.90
# rete_10  100.00  100.00
# re_2     59.34   59.34
# re_5     96.90   96.90
# re_10    100.00  100.00
# te_2     90.23   90.23
# te_5     100.00  100.00
# te_10    100.00  100.00
# proj_2   86.01   86.01
# proj_5   98.40   98.40
# proj_10  99.91   99.91
# re       2.03    2.03
# te       0.01    0.01
