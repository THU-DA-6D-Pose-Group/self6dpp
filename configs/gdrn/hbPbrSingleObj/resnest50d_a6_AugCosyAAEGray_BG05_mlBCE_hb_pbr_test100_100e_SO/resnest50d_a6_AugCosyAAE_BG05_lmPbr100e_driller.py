_base_ = ["./resnest50d_a6_AugCosyAAE_BG05_lmPbr100e_benchvise.py"]

OUTPUT_DIR = "output/gdrn/hb_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/driller"

DATASETS = dict(
    TRAIN=("lm_pbr_driller_train",),
    TEST=("hb_bdp_driller_test100_lmK",),
)
