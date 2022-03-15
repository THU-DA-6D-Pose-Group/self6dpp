_base_ = "./resnest50d_AugCosyAAE_BG05_mlBCE_DoubleMask_visib10_ycbvRealKuwPbr_100e_01_02MasterChefCan.py"
OUTPUT_DIR = (
    "output/gdrn/ycbv/resnest50d_AugCosyAAE_BG05_mlBCE_DoubleMask_visib10_ycbvRealKuwPbr_100e_SO/18_40LargeMarker"
)
DATASETS = dict(TRAIN=("ycbv_040_large_marker_train_real_aligned_Kuw", "ycbv_040_large_marker_train_pbr"))
