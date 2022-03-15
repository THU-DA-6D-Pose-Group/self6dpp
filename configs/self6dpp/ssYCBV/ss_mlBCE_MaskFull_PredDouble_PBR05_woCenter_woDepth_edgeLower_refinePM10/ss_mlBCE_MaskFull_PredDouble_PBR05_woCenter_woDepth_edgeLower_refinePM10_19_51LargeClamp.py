_base_ = "./ss_mlBCE_MaskFull_PredDouble_PBR05_woCenter_woDepth_edgeLower_refinePM10_01_02MasterChefCan.py"
OUTPUT_DIR = (
    "output/self6dpp/ssYCBV/ss_mlBCE_MaskFull_PredDouble_PBR05_woCenter_woDepth_edgeLower_refinePM10/19_51LargeClamp"
)
DATASETS = dict(
    TRAIN=("ycbv_051_large_clamp_train_real_aligned_Kuw",),
    TRAIN2=("ycbv_051_large_clamp_train_pbr",),
)
MODEL = dict(
    WEIGHTS="output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/19_51LargeClamp/model_final_wo_optim-9643daa1.pth"
)
