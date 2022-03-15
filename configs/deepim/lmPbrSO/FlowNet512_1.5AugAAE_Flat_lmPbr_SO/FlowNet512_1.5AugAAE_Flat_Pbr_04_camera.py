_base_ = ["./FlowNet512_1.5AugAAE_Flat_Pbr_01_ape.py"]


OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugAAE_Flat_lmPbr_SO/camera"

DATASETS = dict(TRAIN=("lm_pbr_camera_train",), TEST=("lm_real_camera_test",))

# bbnc7
# objects  camera  Avg(1)
# ad_2     28.04   28.04
# ad_5     84.80   84.80
# ad_10    98.73   98.73
# rete_2   37.84   37.84
# rete_5   98.92   98.92
# rete_10  100.00  100.00
# re_2     37.94   37.94
# re_5     98.92   98.92
# re_10    100.00  100.00
# te_2     99.31   99.31
# te_5     100.00  100.00
# te_10    100.00  100.00
# proj_2   80.88   80.88
# proj_5   99.22   99.22
# proj_10  100.00  100.00
# re       2.38    2.38
# te       0.01    0.01
