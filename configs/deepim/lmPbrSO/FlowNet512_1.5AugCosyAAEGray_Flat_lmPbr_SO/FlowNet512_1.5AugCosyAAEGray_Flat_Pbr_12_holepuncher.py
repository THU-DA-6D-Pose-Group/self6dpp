_base_ = ["./FlowNet512_1.5AugCosyAAEGray_Flat_Pbr_01_ape.py"]


OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_Flat_lmPbr_SO/holepuncher"

DATASETS = dict(TRAIN=("lm_pbr_holepuncher_train",), TEST=("lm_real_holepuncher_test",))

# gu
# objects  holepuncher  Avg(1)
# ad_2     0.76         0.76
# ad_5     7.52         7.52
# ad_10    24.07        24.07
# rete_2   15.51        15.51
# rete_5   90.96        90.96
# rete_10  98.95        98.95
# re_2     40.82        40.82
# re_5     94.58        94.58
# re_10    98.95        98.95
# te_2     40.72        40.72
# te_5     95.53        95.53
# te_10    99.52        99.52
# proj_2   58.14        58.14
# proj_5   98.67        98.67
# proj_10  99.52        99.52
# re       2.93         2.93
# te       0.03         0.03

# init by mlBCE
# objects  holepuncher  Avg(1)
# ad_2     0.67         0.67
# ad_5     7.52         7.52
# ad_10    24.64        24.64
# rete_2   15.41        15.41
# rete_5   91.15        91.15
# rete_10  98.86        98.86
# re_2     40.53        40.53
# re_5     94.77        94.77
# re_10    98.86        98.86
# te_2     42.06        42.06
# te_5     95.62        95.62
# te_10    99.52        99.52
# proj_2   59.37        59.37
# proj_5   98.76        98.76
# proj_10  99.52        99.52
# re       2.78         2.78
# te       0.03         0.03
