_base_ = ["./FlowNet512_1.5AugCosyAAEGray_Flat_Pbr_01_ape.py"]


OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_Flat_lmPbr_SO/duck"

DATASETS = dict(TRAIN=("lm_pbr_duck_train",), TEST=("lm_real_duck_test",))

# bbnc9
# objects  duck    Avg(1)
# ad_2     3.76    3.76
# ad_5     27.14   27.14
# ad_10    61.60   61.60
# rete_2   53.33   53.33
# rete_5   96.71   96.71
# rete_10  100.00  100.00
# re_2     57.65   57.65
# re_5     96.71   96.71
# re_10    100.00  100.00
# te_2     89.11   89.11
# te_5     100.00  100.00
# te_10    100.00  100.00
# proj_2   85.26   85.26
# proj_5   98.22   98.22
# proj_10  99.91   99.91
# re       2.04    2.04
# te       0.01    0.01

# init by mlBCE
# objects  duck    Avg(1)
# ad_2     3.38    3.38
# ad_5     26.85   26.85
# ad_10    61.13   61.13
# rete_2   52.58   52.58
# rete_5   96.34   96.34
# rete_10  100.00  100.00
# re_2     57.09   57.09
# re_5     96.34   96.34
# re_10    100.00  100.00
# te_2     88.73   88.73
# te_5     100.00  100.00
# te_10    100.00  100.00
# proj_2   85.26   85.26
# proj_5   98.22   98.22
# proj_10  99.91   99.91
# re       2.04    2.04
# te       0.01    0.01
