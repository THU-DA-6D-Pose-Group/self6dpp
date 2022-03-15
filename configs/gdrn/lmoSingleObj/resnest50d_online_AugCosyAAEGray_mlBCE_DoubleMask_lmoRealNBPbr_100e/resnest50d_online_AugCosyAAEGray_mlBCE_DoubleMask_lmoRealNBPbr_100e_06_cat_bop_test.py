_base_ = "./resnest50d_online_AugCosyAAEGray_mlBCE_DoubleMask_lmoRealNBPbr_100e_01_ape_bop_test.py"
OUTPUT_DIR = "output/gdrn/lmoRealPbrSO/resnest50d_online_AugCosyAAEGray_mlBCE_DoubleMask_lmoRealNBPbr_100e_SO/cat"
DATASETS = dict(TRAIN=("lmo_pbr_cat_train", "lmo_NoBopTest_cat_train"))
