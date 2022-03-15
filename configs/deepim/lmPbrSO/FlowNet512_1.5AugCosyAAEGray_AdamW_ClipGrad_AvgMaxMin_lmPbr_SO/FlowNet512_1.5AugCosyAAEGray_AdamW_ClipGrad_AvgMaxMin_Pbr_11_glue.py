_base_ = "./FlowNet512_1.5AugCosyAAEGray_AdamW_ClipGrad_AvgMaxMin_Pbr_01_ape.py"
OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AdamW_ClipGrad_AvgMaxMin_lmPbr_SO/glue"
DATASETS = dict(TRAIN=("lm_pbr_glue_train",), TEST=("lm_real_glue_test",))

# bbnc4
# objects  glue   Avg(1)
# ad_2     8.69   8.69
# ad_5     53.86  53.86
# ad_10    88.42  88.42
# rete_2   1.64   1.64
# rete_5   26.35  26.35
# rete_10  71.62  71.62
# re_2     2.12   2.12
# re_5     26.74  26.74
# re_10    71.62  71.62
# te_2     61.49  61.49
# te_5     96.72  96.72
# te_10    99.90  99.90
# proj_2   3.19   3.19
# proj_5   70.85  70.85
# proj_10  99.23  99.23
# re       8.15   8.15
# te       0.02   0.02
