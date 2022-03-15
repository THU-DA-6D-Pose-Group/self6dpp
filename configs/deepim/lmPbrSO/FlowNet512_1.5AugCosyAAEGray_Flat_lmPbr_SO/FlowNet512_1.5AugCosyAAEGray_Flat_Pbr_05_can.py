_base_ = ["./FlowNet512_1.5AugCosyAAEGray_Flat_Pbr_01_ape.py"]


OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_Flat_lmPbr_SO/can"

DATASETS = dict(TRAIN=("lm_pbr_can_train",), TEST=("lm_real_can_test",))

# bbnc10
# objects  can     Avg(1)
# ad_2     17.72   17.72
# ad_5     89.57   89.57
# ad_10    99.80   99.80
# rete_2   97.15   97.15
# rete_5   100.00  100.00
# rete_10  100.00  100.00
# re_2     97.24   97.24
# re_5     100.00  100.00
# re_10    100.00  100.00
# te_2     99.80   99.80
# te_5     100.00  100.00
# te_10    100.00  100.00
# proj_2   45.77   45.77
# proj_5   98.62   98.62
# proj_10  100.00  100.00
# re       0.92    0.92
# te       0.01    0.01

# init by mlBCE
# objects  can     Avg(1)
# ad_2     17.42   17.42
# ad_5     89.07   89.07
# ad_10    99.90   99.90
# rete_2   97.05   97.05
# rete_5   100.00  100.00
# rete_10  100.00  100.00
# re_2     97.24   97.24
# re_5     100.00  100.00
# re_10    100.00  100.00
# te_2     99.80   99.80
# te_5     100.00  100.00
# te_10    100.00  100.00
# proj_2   45.18   45.18
# proj_5   98.62   98.62
# proj_10  100.00  100.00
# re       0.92    0.92
# te       0.01    0.01
