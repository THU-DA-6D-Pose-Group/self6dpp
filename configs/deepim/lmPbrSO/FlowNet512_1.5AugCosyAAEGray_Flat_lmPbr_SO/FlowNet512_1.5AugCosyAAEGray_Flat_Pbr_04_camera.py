_base_ = ["./FlowNet512_1.5AugCosyAAEGray_Flat_Pbr_01_ape.py"]


OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_Flat_lmPbr_SO/camera"

DATASETS = dict(TRAIN=("lm_pbr_camera_train",), TEST=("lm_real_camera_test",))

# bbnc7
# objects  camera  Avg(1)
# ad_2     26.86   26.86
# ad_5     84.41   84.41
# ad_10    99.12   99.12
# rete_2   40.88   40.88
# rete_5   99.12   99.12
# rete_10  100.00  100.00
# re_2     40.98   40.98
# re_5     99.12   99.12
# re_10    100.00  100.00
# te_2     99.51   99.51
# te_5     100.00  100.00
# te_10    100.00  100.00
# proj_2   81.76   81.76
# proj_5   99.31   99.31
# proj_10  100.00  100.00
# re       2.34    2.34
# te       0.01    0.01

# init by mlBCE
# objects  camera  Avg(1)
# ad_2     28.43   28.43
# ad_5     84.31   84.31
# ad_10    99.22   99.22
# rete_2   40.20   40.20
# rete_5   99.02   99.02
# rete_10  100.00  100.00
# re_2     40.29   40.29
# re_5     99.02   99.02
# re_10    100.00  100.00
# te_2     99.51   99.51
# te_5     100.00  100.00
# te_10    100.00  100.00
# proj_2   82.06   82.06
# proj_5   99.31   99.31
# proj_10  100.00  100.00
# re       2.34    2.34
# te       0.01    0.01
