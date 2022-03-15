_base_ = "./FlowNet512_1.5AugCosyAAEGray_Aggressive_Flat_Pbr_01_ape.py"
OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_Aggressive_Flat_lmPbr_SO/camera"
DATASETS = dict(TRAIN=("lm_pbr_camera_train",), TEST=("lm_real_camera_test",))

# bbnc7
# objects  camera  Avg(1)
# ad_2     30.20   30.20
# ad_5     86.57   86.57
# ad_10    99.02   99.02
# rete_2   41.67   41.67
# rete_5   97.55   97.55
# rete_10  100.00  100.00
# re_2     41.67   41.67
# re_5     97.55   97.55
# re_10    100.00  100.00
# te_2     99.51   99.51
# te_5     100.00  100.00
# te_10    100.00  100.00
# proj_2   81.96   81.96
# proj_5   99.22   99.22
# proj_10  100.00  100.00
# re       2.35    2.35
# te       0.00    0.00
