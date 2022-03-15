_base_ = ["./resnest50d_a6_AugCosyAAE_BG05_HBReal_200e_benchvise.py"]

OUTPUT_DIR = "output/gdrn/hbSO/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_HBReal_200e/phone"

DATASETS = dict(
    TRAIN=("hb_bdp_phone_train_lmK",),
    TEST=("hb_bdp_phone_test_lmK",),
)

MODEL = dict(
    WEIGHTS="output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/phone/model_final_wo_optim-525a29f8.pth",
)
