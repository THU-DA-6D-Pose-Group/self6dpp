_base_ = ["./FlowNet512_1.5AugCosyAAEGray_Flat_Pbr_01_ape.py"]


OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_Flat_lmPbr_SO/driller"

DATASETS = dict(TRAIN=("lm_pbr_driller_train",), TEST=("lm_real_driller_test",))

# bbnc10
# objects  driller  Avg(1)
# ad_2     51.44    51.44
# ad_5     94.75    94.75
# ad_10    100.00   100.00
# rete_2   89.79    89.79
# rete_5   99.90    99.90
# rete_10  100.00   100.00
# re_2     89.99    89.99
# re_5     99.90    99.90
# re_10    100.00   100.00
# te_2     99.80    99.80
# te_5     100.00   100.00
# te_10    100.00   100.00
# proj_2   86.52    86.52
# proj_5   98.91    98.91
# proj_10  100.00   100.00
# re       1.20     1.20
# te       0.01     0.01

# init by mlBCE
# objects  driller  Avg(1)
# ad_2     51.14    51.14
# ad_5     94.35    94.35
# ad_10    100.00   100.00
# rete_2   89.59    89.59
# rete_5   99.90    99.90
# rete_10  100.00   100.00
# re_2     89.79    89.79
# re_5     99.90    99.90
# re_10    100.00   100.00
# te_2     99.80    99.80
# te_5     100.00   100.00
# te_10    100.00   100.00
# proj_2   86.62    86.62
# proj_5   98.91    98.91
# proj_10  100.00   100.00
# re       1.21     1.21
# te       0.01     0.01
