_base_ = "./FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Blender_01_ape.py"
OUTPUT_DIR = "output/deepim/lmCropBlenderSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmCropBlender_SO/benchvise"
DATASETS = dict(TRAIN=("lm_blender_benchvise_train",), TEST=("lm_crop_benchvise_test",))

# bbnc5
# iter0
# objects  benchvise  Avg(1)
# ad_2     11.52      11.52
# ad_5     55.97      55.97
# ad_10    90.53      90.53
# rete_2   36.21      36.21
# rete_5   98.35      98.35
# rete_10  100.00     100.00
# re_2     46.50      46.50
# re_5     98.35      98.35
# re_10    100.00     100.00
# te_2     83.54      83.54
# te_5     100.00     100.00
# te_10    100.00     100.00
# proj_2   57.61      57.61
# proj_5   99.18      99.18
# proj_10  100.00     100.00
# re       2.24       2.24
# te       0.01       0.01

# iter4
# objects  benchvise  Avg(1)
# ad_2     7.41       7.41
# ad_5     33.33      33.33
# ad_10    82.30      82.30
# rete_2   30.04      30.04
# rete_5   98.35      98.35
# rete_10  100.00     100.00
# re_2     48.15      48.15
# re_5     98.35      98.35
# re_10    100.00     100.00
# te_2     65.43      65.43
# te_5     100.00     100.00
# te_10    100.00     100.00
# proj_2   66.67      66.67
# proj_5   99.59      99.59
# proj_10  100.00     100.00
# re       2.21       2.21
# te       0.02       0.02
