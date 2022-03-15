_base_ = "./FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Blender_01_ape.py"
OUTPUT_DIR = "output/deepim/lmCropBlenderSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmCropBlender_SO/holepuncher"
DATASETS = dict(TRAIN=("lm_blender_holepuncher_train",), TEST=("lm_crop_holepuncher_test",))

# bbnc9
# iter0
# objects  holepuncher  Avg(1)
# ad_2     0.00         0.00
# ad_5     0.00         0.00
# ad_10    0.00         0.00
# rete_2   0.00         0.00
# rete_5   23.39        23.39
# rete_10  52.82        52.82
# re_2     4.84         4.84
# re_5     40.73        40.73
# re_10    78.23        78.23
# te_2     1.21         1.21
# te_5     33.87        33.87
# te_10    58.47        58.47
# proj_2   0.00         0.00
# proj_5   4.03         4.03
# proj_10  33.87        33.87
# re       7.44         7.44
# te       0.09         0.09
#
# iter4
# objects  holepuncher  Avg(1)
# ad_2     0.00         0.00
# ad_5     0.00         0.00
# ad_10    5.65         5.65
# rete_2   8.06         8.06
# rete_5   71.77        71.77
# rete_10  95.97        95.97
# re_2     22.18        22.18
# re_5     77.82        77.82
# re_10    96.37        96.37
# te_2     23.39        23.39
# te_5     88.71        88.71
# te_10    99.19        99.19
# proj_2   0.00         0.00
# proj_5   7.26         7.26
# proj_10  35.48        35.48
# re       4.20         4.20
# te       0.03         0.03
