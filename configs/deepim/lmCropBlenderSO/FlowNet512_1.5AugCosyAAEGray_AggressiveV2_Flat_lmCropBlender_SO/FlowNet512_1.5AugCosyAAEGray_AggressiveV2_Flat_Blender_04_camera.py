_base_ = "./FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Blender_01_ape.py"
OUTPUT_DIR = "output/deepim/lmCropBlenderSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmCropBlender_SO/camera"
DATASETS = dict(TRAIN=("lm_blender_camera_train",), TEST=("lm_crop_camera_test",))

# bbnc5
# iter0
# objects  camera  Avg(1)
# ad_2     4.56    4.56
# ad_5     23.24   23.24
# ad_10    43.57   43.57
# rete_2   14.94   14.94
# rete_5   63.07   63.07
# rete_10  90.46   90.46
# re_2     21.16   21.16
# re_5     75.52   75.52
# re_10    96.27   96.27
# te_2     47.30   47.30
# te_5     71.78   71.78
# te_10    93.78   93.78
# proj_2   26.97   26.97
# proj_5   84.23   84.23
# proj_10  99.17   99.17
# re       3.98    3.98
# te       0.03    0.03

# iter4
# objects  camera  Avg(1)
# ad_2     2.07    2.07
# ad_5     28.63   28.63
# ad_10    80.91   80.91
# rete_2   43.15   43.15
# rete_5   98.34   98.34
# rete_10  99.59   99.59
# re_2     44.81   44.81
# re_5     98.34   98.34
# re_10    99.59   99.59
# te_2     91.70   91.70
# te_5     100.00  100.00
# te_10    100.00  100.00
# proj_2   77.18   77.18
# proj_5   99.59   99.59
# proj_10  99.59   99.59
# re       2.41    2.41
# te       0.01    0.01
