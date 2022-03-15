_base_ = "./resnest50d_AugCosyAAE_BG05_mlBCE_DoubleMask_visib10_ycbvRealKuwPbr_100e_01_02MasterChefCan_bop_test.py"
OUTPUT_DIR = (
    "output/gdrn/ycbv/resnest50d_AugCosyAAE_BG05_mlBCE_DoubleMask_visib10_ycbvRealKuwPbr_100e_SO/12_21BleachCleanser"
)
DATASETS = dict(TRAIN=("ycbv_021_bleach_cleanser_train_real_aligned_Kuw", "ycbv_021_bleach_cleanser_train_pbr"))
