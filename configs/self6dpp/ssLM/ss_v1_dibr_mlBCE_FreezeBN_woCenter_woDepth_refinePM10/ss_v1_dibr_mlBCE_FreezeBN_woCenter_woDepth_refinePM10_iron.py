_base_ = "./ss_v1_dibr_mlBCE_FreezeBN_woCenter_woDepth_refinePM10_ape.py"
OUTPUT_DIR = "output/self6dpp/ssLM/ss_v1_dibr_mlBCE_FreezeBN_woCenter_woDepth_refinePM10/iron"
DATASETS = dict(
    TRAIN=("lm_real_iron_train",), TRAIN2=("lm_pbr_iron_train",), TRAIN2_RATIO=0.0, TEST=("lm_real_iron_test",)
)
MODEL = dict(
    WEIGHTS="output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/iron/model_final_wo_optim-025a740e.pth"
)
