_base_ = "./FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_01_ape.py"
OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/benchvise"
DATASETS = dict(TRAIN=("lm_pbr_benchvise_train",), TEST=("lm_real_benchvise_test",))

# ./core/deepim/test_deepim.sh configs/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_02_benchvise.py 1 output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/benchvise/model_final_wo_optim-956eb2eb.pth
# ./core/deepim/save_deepim.sh configs/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_02_benchvise.py 1 output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/benchvise/model_final_wo_optim-956eb2eb.pth
# bbnc7
# objects  benchvise  Avg(1)
# ad_2     10.09      10.09
# ad_5     45.78      45.78
# ad_10    93.11      93.11
# rete_2   46.56      46.56
# rete_5   99.71      99.71
# rete_10  100.00     100.00
# re_2     61.49      61.49
# re_5     99.71      99.71
# re_10    100.00     100.00
# te_2     80.02      80.02
# te_5     100.00     100.00
# te_10    100.00     100.00
# proj_2   75.17      75.17
# proj_5   99.32      99.32
# proj_10  100.00     100.00
# re       1.85       1.85
# te       0.01       0.01
