_base_ = "./FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Blender_01_ape.py"
OUTPUT_DIR = "output/deepim/lmCropBlenderSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmCropBlender_SO/can"
DATASETS = dict(TRAIN=("lm_blender_can_train",), TEST=("lm_crop_can_test",))

# bbnc5
# iter0
# objects  can     Avg(1)
# ad_2     17.92   17.92
# ad_5     79.17   79.17
# ad_10    98.33   98.33
# rete_2   57.92   57.92
# rete_5   98.33   98.33
# rete_10  100.00  100.00
# re_2     57.92   57.92
# re_5     98.33   98.33
# re_10    100.00  100.00
# te_2     97.92   97.92
# te_5     100.00  100.00
# te_10    100.00  100.00
# proj_2   20.42   20.42
# proj_5   96.67   96.67
# proj_10  100.00  100.00
# re       1.96    1.96
# te       0.01    0.01

# iter4
# objects  can     Avg(1)
# ad_2     27.92   27.92
# ad_5     91.25   91.25
# ad_10    99.58   99.58
# rete_2   88.75   88.75
# rete_5   99.58   99.58
# rete_10  100.00  100.00
# re_2     88.75   88.75
# re_5     99.58   99.58
# re_10    100.00  100.00
# te_2     99.58   99.58
# te_5     100.00  100.00
# te_10    100.00  100.00
# proj_2   33.75   33.75
# proj_5   96.67   96.67
# proj_10  100.00  100.00
# re       1.34    1.34
# te       0.01    0.01
