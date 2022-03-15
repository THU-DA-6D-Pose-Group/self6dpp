_base_ = "./resnest50d_AugCosyAAE_BG05_mlBCE_DoubleMask_visib10_ycbvRealKuwPbr_100e_01_02MasterChefCan.py"
OUTPUT_DIR = (
    "output/gdrn/ycbv/resnest50d_AugCosyAAE_BG05_mlBCE_DoubleMask_visib10_ycbvRealKuwPbr_100e_SO/20_52ExtraLargeClamp"
)
DATASETS = dict(TRAIN=("ycbv_052_extra_large_clamp_train_real_aligned_Kuw", "ycbv_052_extra_large_clamp_train_pbr"))
