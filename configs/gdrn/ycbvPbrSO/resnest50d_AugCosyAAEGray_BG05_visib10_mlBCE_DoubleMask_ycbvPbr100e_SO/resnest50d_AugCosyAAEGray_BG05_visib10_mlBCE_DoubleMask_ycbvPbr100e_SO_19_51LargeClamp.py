_base_ = "./resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_01_02MasterChefCan.py"
OUTPUT_DIR = (
    "output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/19_51LargeClamp"
)
DATASETS = dict(TRAIN=("ycbv_051_large_clamp_train_pbr",))
