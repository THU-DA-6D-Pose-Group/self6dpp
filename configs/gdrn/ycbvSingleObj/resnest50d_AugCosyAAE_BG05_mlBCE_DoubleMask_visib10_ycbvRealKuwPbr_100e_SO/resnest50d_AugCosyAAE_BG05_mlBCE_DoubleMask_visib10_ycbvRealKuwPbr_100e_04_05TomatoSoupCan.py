_base_ = "./resnest50d_AugCosyAAE_BG05_mlBCE_DoubleMask_visib10_ycbvRealKuwPbr_100e_01_02MasterChefCan.py"
OUTPUT_DIR = (
    "output/gdrn/ycbv/resnest50d_AugCosyAAE_BG05_mlBCE_DoubleMask_visib10_ycbvRealKuwPbr_100e_SO/04_05TomatoSoupCan"
)
DATASETS = dict(TRAIN=("ycbv_005_tomato_soup_can_train_real_aligned_Kuw", "ycbv_005_tomato_soup_can_train_pbr"))
