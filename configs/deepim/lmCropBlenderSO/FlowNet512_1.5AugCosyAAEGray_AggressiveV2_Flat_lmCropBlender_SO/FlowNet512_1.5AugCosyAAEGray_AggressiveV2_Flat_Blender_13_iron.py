_base_ = "./FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Blender_01_ape.py"
OUTPUT_DIR = "output/deepim/lmCropBlenderSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmCropBlender_SO/iron"
DATASETS = dict(TRAIN=("lm_blender_iron_train",), TEST=("lm_crop_iron_test",))

# bbnc9
# iter0
# objects  iron    Avg(1)
# ad_2     5.19    5.19
# ad_5     67.97   67.97
# ad_10    94.37   94.37
# rete_2   32.90   32.90
# rete_5   87.45   87.45
# rete_10  95.67   95.67
# re_2     35.06   35.06
# re_5     87.88   87.88
# re_10    95.67   95.67
# te_2     87.45   87.45
# te_5     99.57   99.57
# te_10    100.00  100.00
# proj_2   6.06    6.06
# proj_5   61.90   61.90
# proj_10  99.13   99.13
# re       3.25    3.25
# te       0.01    0.01
#
# iter4
# objects  iron    Avg(1)
# ad_2     6.49    6.49
# ad_5     80.95   80.95
# ad_10    99.13   99.13
# rete_2   43.29   43.29
# rete_5   98.27   98.27
# rete_10  100.00  100.00
# re_2     44.59   44.59
# re_5     98.27   98.27
# re_10    100.00  100.00
# te_2     94.81   94.81
# te_5     100.00  100.00
# te_10    100.00  100.00
# proj_2   10.82   10.82
# proj_5   86.58   86.58
# proj_10  100.00  100.00
# re       2.29    2.29
# te       0.01    0.01
