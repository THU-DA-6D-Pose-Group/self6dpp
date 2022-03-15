_base_ = "./FlowNet512_1.5AugCosyAAEGray_Aggressive_Flat_Pbr_01_ape.py"
OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_Aggressive_Flat_lmPbr_SO/eggbox"
DATASETS = dict(TRAIN=("lm_pbr_eggbox_train",), TEST=("lm_real_eggbox_test",))

# rl1
# objects  eggbox  Avg(1)
# ad_2     13.90   13.90
# ad_5     49.39   49.39
# ad_10    93.43   93.43
# rete_2   41.50   41.50
# rete_5   95.12   95.12
# rete_10  99.81   99.81
# re_2     58.50   58.50
# re_5     95.49   95.49
# re_10    99.81   99.81
# te_2     58.31   58.31
# te_5     98.97   98.97
# te_10    99.91   99.91
# proj_2   58.50   58.50
# proj_5   98.59   98.59
# proj_10  99.91   99.91
# re       2.13    2.13
# te       0.02    0.02
