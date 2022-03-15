_base_ = "./FlowNet512_1.5AugCosyAAEGray_Aggressive_Flat_Pbr_01_ape.py"
OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_Aggressive_Flat_lmPbr_SO/holepuncher"
DATASETS = dict(TRAIN=("lm_pbr_holepuncher_train",), TEST=("lm_real_holepuncher_test",))

# rl1
# objects  holepuncher  Avg(1)
# ad_2     0.38         0.38
# ad_5     6.37         6.37
# ad_10    23.12        23.12
# rete_2   14.94        14.94
# rete_5   90.29        90.29
# rete_10  98.95        98.95
# re_2     40.15        40.15
# re_5     93.91        93.91
# re_10    98.95        98.95
# te_2     39.58        39.58
# te_5     95.72        95.72
# te_10    99.52        99.52
# proj_2   54.04        54.04
# proj_5   98.76        98.76
# proj_10  99.52        99.52
# re       2.81         2.81
# te       0.03         0.03
