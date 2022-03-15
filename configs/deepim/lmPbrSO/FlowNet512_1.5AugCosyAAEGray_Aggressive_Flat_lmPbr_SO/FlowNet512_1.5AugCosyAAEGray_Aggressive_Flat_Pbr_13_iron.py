_base_ = "./FlowNet512_1.5AugCosyAAEGray_Aggressive_Flat_Pbr_01_ape.py"
OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_Aggressive_Flat_lmPbr_SO/iron"
DATASETS = dict(TRAIN=("lm_pbr_iron_train",), TEST=("lm_real_iron_test",))

# rl1
# objects  iron    Avg(1)
# ad_2     10.42   10.42
# ad_5     93.97   93.97
# ad_10    100.00  100.00
# rete_2   38.61   38.61
# rete_5   99.69   99.69
# rete_10  100.00  100.00
# re_2     38.61   38.61
# re_5     99.69   99.69
# re_10    100.00  100.00
# te_2     99.69   99.69
# te_5     100.00  100.00
# te_10    100.00  100.00
# proj_2   9.40    9.40
# proj_5   89.68   89.68
# proj_10  100.00  100.00
# re       2.35    2.35
# te       0.01    0.01
