_base_ = "./ss_mlBCE_MaskFull_PredDouble_PBR05_woCenter_woDepth_edgeLower_refinePM10_01_02MasterChefCan.py"
OUTPUT_DIR = (
    "output/self6dpp/ssYCBV/ss_mlBCE_MaskFull_PredDouble_PBR05_woCenter_woDepth_edgeLower_refinePM10/11_19PitcherBase"
)
DATASETS = dict(
    TRAIN=("ycbv_019_pitcher_base_train_real_aligned_Kuw",),
    TRAIN2=("ycbv_019_pitcher_base_train_pbr",),
)
MODEL = dict(
    WEIGHTS="output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/11_19PitcherBase/model_final_wo_optim-f052dafc.pth"
)
