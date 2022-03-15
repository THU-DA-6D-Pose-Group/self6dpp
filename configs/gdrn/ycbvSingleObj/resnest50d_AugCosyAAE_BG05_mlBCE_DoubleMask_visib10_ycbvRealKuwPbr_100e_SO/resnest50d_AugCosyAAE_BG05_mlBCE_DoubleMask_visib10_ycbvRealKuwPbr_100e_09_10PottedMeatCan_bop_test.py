_base_ = "./resnest50d_AugCosyAAE_BG05_mlBCE_DoubleMask_visib10_ycbvRealKuwPbr_100e_01_02MasterChefCan_bop_test.py"
OUTPUT_DIR = (
    "output/gdrn/ycbv/resnest50d_AugCosyAAE_BG05_mlBCE_DoubleMask_visib10_ycbvRealKuwPbr_100e_SO/09_10PottedMeatCan"
)
DATASETS = dict(TRAIN=("ycbv_010_potted_meat_can_train_real_aligned_Kuw", "ycbv_010_potted_meat_can_train_pbr"))
