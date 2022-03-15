_base_ = "./FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Blender_01_ape.py"
OUTPUT_DIR = "output/deepim/lmCropBlenderSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmCropBlender_SO/duck"
DATASETS = dict(TRAIN=("lm_blender_duck_train",), TEST=("lm_crop_duck_test",))

# bbnc9
# iter0
# objects  duck   Avg(1)
# ad_2     0.00   0.00
# ad_5     0.00   0.00
# ad_10    0.40   0.40
# rete_2   0.80   0.80
# rete_5   11.16  11.16
# rete_10  25.90  25.90
# re_2     3.59   3.59
# re_5     17.13  17.13
# re_10    32.67  32.67
# te_2     0.80   0.80
# te_5     15.54  15.54
# te_10    28.69  28.69
# proj_2   0.80   0.80
# proj_5   24.30  24.30
# proj_10  43.82  43.82
# re       39.39  39.39
# te       0.24   0.24

# iter4
# objects  duck   Avg(1)
# ad_2     3.19   3.19
# ad_5     13.55  13.55
# ad_10    30.28  30.28
# rete_2   20.32  20.32
# rete_5   53.78  53.78
# rete_10  65.34  65.34
# re_2     21.12  21.12
# re_5     56.18  56.18
# re_10    65.34  65.34
# te_2     52.99  52.99
# te_5     62.15  62.15
# te_10    70.92  70.92
# proj_2   49.80  49.80
# proj_5   65.34  65.34
# proj_10  71.71  71.71
# re       32.32  32.32
# te       0.10   0.10
