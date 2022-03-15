_base_ = ["./ss_dibr_mlBCE_FreezeBN_woCenter_refinePM10_10e_train180_benchvise.py"]

# refiner_cfg_path = "configs/_base_/self6dpp_refiner_base.py"
OUTPUT_DIR = "output/self6dpp/ssHB/ss_dibr_mlBCE_FreezeBN_woCenter_refinePM10_10e_train180/driller"

DATASETS = dict(
    TRAIN=("hb_bdp_driller_train180_lmK",),  # real data
    TRAIN2=("lm_pbr_driller_train",),  # synthetic data
    TEST=("hb_bdp_driller_test100_lmK",),
)

RENDERER = dict(DIFF_RENDERER="DIBR")  # DIBR | DIBR

MODEL = dict(
    # synthetically trained model
    WEIGHTS="output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/driller/model_final_wo_optim-4cfc7d64.pth",
)
