_base_ = "./FlowNet512_1.5AugCosyAAEGray_AdamW_ClipGrad_AvgMaxMin_Pbr_01_ape.py"
OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AdamW_ClipGrad_AvgMaxMin_lmPbr_SO/holepuncher"
DATASETS = dict(TRAIN=("lm_pbr_holepuncher_train",), TEST=("lm_real_holepuncher_test",))

# rl3
# objects  holepuncher  Avg(1)
# ad_2     0.29         0.29
# ad_5     8.18         8.18
# ad_10    24.55        24.55
# rete_2   17.22        17.22
# rete_5   91.91        91.91
# rete_10  99.43        99.43
# re_2     44.62        44.62
# re_5     95.24        95.24
# re_10    99.43        99.43
# te_2     44.53        44.53
# te_5     96.00        96.00
# te_10    99.52        99.52
# proj_2   58.52        58.52
# proj_5   98.76        98.76
# proj_10  99.52        99.52
# re       2.81         2.81
# te       0.02         0.02
