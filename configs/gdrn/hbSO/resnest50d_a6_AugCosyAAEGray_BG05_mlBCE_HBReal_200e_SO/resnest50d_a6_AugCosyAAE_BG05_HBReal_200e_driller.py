_base_ = ["./resnest50d_a6_AugCosyAAE_BG05_HBReal_200e_benchvise.py"]

OUTPUT_DIR = "output/gdrn/hbSO/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_HBReal_200e/driller"

DATASETS = dict(
    TRAIN=("hb_bdp_driller_train_lmK",),
    TEST=("hb_bdp_driller_test_lmK",),
)

MODEL = dict(
    WEIGHTS="output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/driller/model_final_wo_optim-4cfc7d64.pth",
)
