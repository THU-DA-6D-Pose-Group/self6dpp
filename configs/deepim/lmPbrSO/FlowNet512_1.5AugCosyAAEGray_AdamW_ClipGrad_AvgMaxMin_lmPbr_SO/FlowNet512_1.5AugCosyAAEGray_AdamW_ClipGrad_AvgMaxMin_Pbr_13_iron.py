_base_ = "./FlowNet512_1.5AugCosyAAEGray_AdamW_ClipGrad_AvgMaxMin_Pbr_01_ape.py"
OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AdamW_ClipGrad_AvgMaxMin_lmPbr_SO/iron"
DATASETS = dict(TRAIN=("lm_pbr_iron_train",), TEST=("lm_real_iron_test",))

# rl3
# objects  iron    Avg(1)
# ad_2     13.28   13.28
# ad_5     93.16   93.16
# ad_10    100.00  100.00
# rete_2   40.14   40.14
# rete_5   99.90   99.90
# rete_10  100.00  100.00
# re_2     40.65   40.65
# re_5     99.90   99.90
# re_10    100.00  100.00
# te_2     99.39   99.39
# te_5     100.00  100.00
# te_10    100.00  100.00
# proj_2   9.60    9.60
# proj_5   89.48   89.48
# proj_10  100.00  100.00
# re       2.26    2.26
# te       0.01    0.01
