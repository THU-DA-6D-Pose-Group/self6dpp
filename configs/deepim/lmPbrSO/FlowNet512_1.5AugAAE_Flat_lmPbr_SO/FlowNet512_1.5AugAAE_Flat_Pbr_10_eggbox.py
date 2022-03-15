_base_ = ["./FlowNet512_1.5AugAAE_Flat_Pbr_01_ape.py"]


OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugAAE_Flat_lmPbr_SO/eggbox"

DATASETS = dict(TRAIN=("lm_pbr_eggbox_train",), TEST=("lm_real_eggbox_test",))

# rl1
# objects  eggbox  Avg(1)
# ad_2     12.77   12.77
# ad_5     53.62   53.62
# ad_10    94.84   94.84
# rete_2   43.29   43.29
# rete_5   97.09   97.09
# rete_10  100.00  100.00
# re_2     56.53   56.53
# re_5     97.46   97.46
# re_10    100.00  100.00
# te_2     62.91   62.91
# te_5     99.53   99.53
# te_10    100.00  100.00
# proj_2   62.72   62.72
# proj_5   98.87   98.87
# proj_10  100.00  100.00
# re       2.03    2.03
# te       0.02    0.02
