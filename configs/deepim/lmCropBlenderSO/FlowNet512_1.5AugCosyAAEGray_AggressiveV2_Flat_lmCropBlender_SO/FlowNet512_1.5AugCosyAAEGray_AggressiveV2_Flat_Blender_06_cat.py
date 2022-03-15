_base_ = "./FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Blender_01_ape.py"
OUTPUT_DIR = "output/deepim/lmCropBlenderSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmCropBlender_SO/cat"
DATASETS = dict(TRAIN=("lm_blender_cat_train",), TEST=("lm_crop_cat_test",))

# iter0
# objects  cat    Avg(1)
# ad_2     2.97   2.97
# ad_5     22.46  22.46
# ad_10    35.17  35.17
# rete_2   22.46  22.46
# rete_5   50.42  50.42
# rete_10  66.53  66.53
# re_2     24.58  24.58
# re_5     66.53  66.53
# re_10    84.32  84.32
# te_2     38.56  38.56
# te_5     53.39  53.39
# te_10    69.49  69.49
# proj_2   16.95  16.95
# proj_5   73.73  73.73
# proj_10  91.95  91.95
# re       9.27   9.27
# te       0.07   0.07

# iter4
# objects  cat    Avg(1)
# ad_2     11.44  11.44
# ad_5     55.08  55.08
# ad_10    83.47  83.47
# rete_2   67.37  67.37
# rete_5   93.64  93.64
# rete_10  96.61  96.61
# re_2     70.76  70.76
# re_5     93.64  93.64
# re_10    96.61  96.61
# te_2     90.68  90.68
# te_5     97.03  97.03
# te_10    97.03  97.03
# proj_2   79.24  79.24
# proj_5   95.76  95.76
# proj_10  96.61  96.61
# re       5.52   5.52
# te       0.02   0.02
