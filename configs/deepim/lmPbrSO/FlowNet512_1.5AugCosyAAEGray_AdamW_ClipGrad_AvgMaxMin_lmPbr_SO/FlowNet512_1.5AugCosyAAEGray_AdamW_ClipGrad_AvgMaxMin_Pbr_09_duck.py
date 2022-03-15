_base_ = "./FlowNet512_1.5AugCosyAAEGray_AdamW_ClipGrad_AvgMaxMin_Pbr_01_ape.py"
OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AdamW_ClipGrad_AvgMaxMin_lmPbr_SO/duck"
DATASETS = dict(TRAIN=("lm_pbr_duck_train",), TEST=("lm_real_duck_test",))


# bbnc4
# objects  duck    Avg(1)
# ad_2     2.82    2.82
# ad_5     22.63   22.63
# ad_10    53.99   53.99
# rete_2   51.92   51.92
# rete_5   98.40   98.40
# rete_10  100.00  100.00
# re_2     57.93   57.93
# re_5     98.40   98.40
# re_10    100.00  100.00
# te_2     88.17   88.17
# te_5     100.00  100.00
# te_10    100.00  100.00
# proj_2   85.63   85.63
# proj_5   98.22   98.22
# proj_10  99.91   99.91
# re       2.01    2.01
# te       0.01    0.01
