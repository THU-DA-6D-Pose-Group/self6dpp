_base_ = "./FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_01_ape.py"
OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/holepuncher"
DATASETS = dict(TRAIN=("lm_pbr_holepuncher_train",), TEST=("lm_real_holepuncher_test",))

# bbnc7
# objects  holepuncher  Avg(1)
# ad_2     0.29         0.29
# ad_5     9.90         9.90
# ad_10    32.06        32.06
# rete_2   17.22        17.22
# rete_5   93.43        93.43
# rete_10  99.24        99.24
# re_2     37.39        37.39
# re_5     95.05        95.05
# re_10    99.33        99.33
# te_2     52.24        52.24
# te_5     97.81        97.81
# te_10    99.52        99.52
# proj_2   53.38        53.38
# proj_5   98.86        98.86
# proj_10  99.52        99.52
# re       2.83         2.83
# te       0.02         0.02
