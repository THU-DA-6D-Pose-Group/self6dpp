_base_ = ["./FlowNet512_1.5AugAAE_Flat_Pbr_01_ape.py"]


OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugAAE_Flat_lmPbr_SO/holepuncher"

DATASETS = dict(TRAIN=("lm_pbr_holepuncher_train",), TEST=("lm_real_holepuncher_test",))

# rl1
# objects  holepuncher  Avg(1)
# ad_2     0.95         0.95
# ad_5     9.99         9.99
# ad_10    24.17        24.17
# rete_2   13.61        13.61
# rete_5   87.44        87.44
# rete_10  99.14        99.14
# re_2     36.92        36.92
# re_5     90.49        90.49
# re_10    99.14        99.14
# te_2     41.86        41.86
# te_5     96.00        96.00
# te_10    99.52        99.52
# proj_2   57.94        57.94
# proj_5   98.95        98.95
# proj_10  99.52        99.52
# re       3.21         3.21
# te       0.03         0.03
