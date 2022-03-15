_base_ = ["ss_mlBCE_MaskFull_PredFull_lr1e_5_lower_woCenter_woDepth_refinePM10_01_ape.py"]

OUTPUT_DIR = (
    "output/self6dpp/ssLMO/ss_mlBCE_MaskFull_PredFull_lr1e_5_lower_woCenter_woDepth_refinePM10_lmoNoBopTest/cat"
)

DATASETS = dict(TRAIN=("lmo_NoBopTest_cat_train",), TRAIN2=("lmo_pbr_cat_train",))  # real data  # synthetic data

MODEL = dict(
    # synthetically trained model
    WEIGHTS="output/gdrn/lmoPbrSO/resnest50d_online_AugCosyAAEGray_mlBCE_DoubleMask_lmo_pbr_100e/cat/model_final_wo_optim-9931aeed.pth"
)
