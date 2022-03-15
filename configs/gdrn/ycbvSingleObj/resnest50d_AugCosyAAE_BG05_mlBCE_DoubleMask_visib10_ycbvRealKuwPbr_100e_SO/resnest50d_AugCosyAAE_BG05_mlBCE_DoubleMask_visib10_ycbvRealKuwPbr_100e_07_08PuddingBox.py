_base_ = "./resnest50d_AugCosyAAE_BG05_mlBCE_DoubleMask_visib10_ycbvRealKuwPbr_100e_01_02MasterChefCan.py"
OUTPUT_DIR = (
    "output/gdrn/ycbv/resnest50d_AugCosyAAE_BG05_mlBCE_DoubleMask_visib10_ycbvRealKuwPbr_100e_SO/07_08PuddingBox"
)
DATASETS = dict(TRAIN=("ycbv_008_pudding_box_train_real_aligned_Kuw", "ycbv_008_pudding_box_train_pbr"))
