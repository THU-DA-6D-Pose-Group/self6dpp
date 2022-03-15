_base_ = "./FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_01_ape_lmo_test.py"
OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/holepuncher"
DATASETS = dict(TRAIN=("lm_pbr_holepuncher_train",), TEST=("lmo_test",))
