_base_ = ["./FlowNet512_1.5AugAAE_Flat_Pbr_01_ape.py"]


OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugAAE_Flat_lmPbr_SO/iron"

DATASETS = dict(TRAIN=("lm_pbr_iron_train",), TEST=("lm_real_iron_test",))

# rl1
# objects  iron    Avg(1)
# ad_2     13.89   13.89
# ad_5     96.42   96.42
# ad_10    100.00  100.00
# rete_2   43.00   43.00
# rete_5   99.69   99.69
# rete_10  100.00  100.00
# re_2     43.11   43.11
# re_5     99.69   99.69
# re_10    100.00  100.00
# te_2     99.59   99.59
# te_5     100.00  100.00
# te_10    100.00  100.00
# proj_2   11.95   11.95
# proj_5   91.22   91.22
# proj_10  100.00  100.00
# re       2.21    2.21
# te       0.01    0.01
