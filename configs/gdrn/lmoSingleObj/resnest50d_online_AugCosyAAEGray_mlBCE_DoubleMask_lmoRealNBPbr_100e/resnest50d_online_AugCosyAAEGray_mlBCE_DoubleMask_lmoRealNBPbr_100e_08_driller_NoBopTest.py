_base_ = "./resnest50d_online_AugCosyAAEGray_mlBCE_DoubleMask_lmoRealNBPbr_100e_01_ape_NoBopTest.py"
OUTPUT_DIR = "output/gdrn/lmoRealPbrSO/resnest50d_online_AugCosyAAEGray_mlBCE_DoubleMask_lmoRealNBPbr_100e_SO/driller"
DATASETS = dict(TRAIN=("lmo_pbr_driller_train", "lmo_NoBopTest_driller_train"))
