_base_ = "./ss_mlBCE_MaskFull_PredDouble_PBR05_woCenter_woDepth_edgeLower_refinePM10_01_02MasterChefCan.py"
OUTPUT_DIR = (
    "output/self6dpp/ssYCBV/ss_mlBCE_MaskFull_PredDouble_PBR05_woCenter_woDepth_edgeLower_refinePM10/06_07TunaFishCan"
)
DATASETS = dict(
    TRAIN=("ycbv_007_tuna_fish_can_train_real_aligned_Kuw",),
    TRAIN2=("ycbv_007_tuna_fish_can_train_pbr",),
)
MODEL = dict(
    WEIGHTS="output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/06_07TunaFishCan/model_final_wo_optim-0b376921.pth"
)
