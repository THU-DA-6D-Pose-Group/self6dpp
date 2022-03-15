_base_ = ["./FlowNet512_1.5AugCosyAAEGray_Flat_Pbr_01_ape.py"]


OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_Flat_lmPbr_SO/eggbox"

DATASETS = dict(TRAIN=("lm_pbr_eggbox_train",), TEST=("lm_real_eggbox_test",))

# bbnc9
# objects  eggbox  Avg(1)
# ad_2     10.42   10.42
# ad_5     46.10   46.10
# ad_10    92.86   92.86
# rete_2   5.92    5.92
# rete_5   46.29   46.29
# rete_10  86.57   86.57
# re_2     9.67    9.67
# re_5     46.76   46.76
# re_10    86.57   86.57
# te_2     53.24   53.24
# te_5     99.06   99.06
# te_10    100.00  100.00
# proj_2   38.03   38.03
# proj_5   93.80   93.80
# proj_10  99.44   99.44
# re       6.03    6.03
# te       0.02    0.02


# init by mlBCE
# objects  eggbox  Avg(1)
# ad_2     10.23   10.23
# ad_5     45.73   45.73
# ad_10    93.05   93.05
# rete_2   5.45    5.45
# rete_5   45.16   45.16
# rete_10  86.76   86.76
# re_2     8.73    8.73
# re_5     45.35   45.35
# re_10    86.76   86.76
# te_2     52.77   52.77
# te_5     98.97   98.97
# te_10    100.00  100.00
# proj_2   37.37   37.37
# proj_5   92.77   92.77
# proj_10  99.25   99.25
# re       6.13    6.13
# te       0.02    0.02
