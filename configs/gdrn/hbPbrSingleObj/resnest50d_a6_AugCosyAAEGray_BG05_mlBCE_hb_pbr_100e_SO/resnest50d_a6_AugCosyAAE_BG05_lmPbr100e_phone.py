_base_ = ["./resnest50d_a6_AugCosyAAE_BG05_lmPbr100e_benchvise.py"]

OUTPUT_DIR = "output/gdrn/hb_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/phone"

DATASETS = dict(
    TRAIN=("lm_pbr_phone_train",),
    TEST=("hb_bdp_phone_test_lmK",),
)
