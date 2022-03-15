_base_ = ["./FlowNet512_1.5AugCosyAAEGray_Flat_Pbr_01_ape.py"]


OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_Flat_lmPbr_SO/lamp"

DATASETS = dict(TRAIN=("lm_pbr_lamp_train",), TEST=("lm_real_lamp_test",))

# rl3
# objects  lamp   Avg(1)
# ad_2     0.29   0.29
# ad_5     39.25  39.25
# ad_10    99.33  99.33
# rete_2   40.60  40.60
# rete_5   99.81  99.81
# rete_10  99.90  99.90
# re_2     46.07  46.07
# re_5     99.81  99.81
# re_10    99.90  99.90
# te_2     86.95  86.95
# te_5     99.90  99.90
# te_10    99.90  99.90
# proj_2   18.91  18.91
# proj_5   96.16  96.16
# proj_10  99.90  99.90
# re       2.15   2.15
# te       0.02   0.02

# init by mlBCE
# objects  lamp   Avg(1)
# ad_2     0.29   0.29
# ad_5     39.35  39.35
# ad_10    99.23  99.23
# rete_2   40.50  40.50
# rete_5   99.71  99.71
# rete_10  99.90  99.90
# re_2     46.26  46.26
# re_5     99.71  99.71
# re_10    99.90  99.90
# te_2     86.95  86.95
# te_5     99.90  99.90
# te_10    99.90  99.90
# proj_2   18.91  18.91
# proj_5   96.35  96.35
# proj_10  99.90  99.90
# re       2.17   2.17
# te       0.02   0.02
