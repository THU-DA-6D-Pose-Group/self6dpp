_base_ = "./FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_01_ape.py"
OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/camera"
DATASETS = dict(TRAIN=("lm_pbr_camera_train",), TEST=("lm_real_camera_test",))

# bbnc7
# objects  camera  Avg(1)
# ad_2     25.59   25.59
# ad_5     85.29   85.29
# ad_10    99.12   99.12
# rete_2   40.88   40.88
# rete_5   99.02   99.02
# rete_10  100.00  100.00
# re_2     40.98   40.98
# re_5     99.02   99.02
# re_10    100.00  100.00
# te_2     99.71   99.71
# te_5     100.00  100.00
# te_10    100.00  100.00
# proj_2   82.55   82.55
# proj_5   99.12   99.12
# proj_10  100.00  100.00
# re       2.32    2.32
# te       0.01    0.01
