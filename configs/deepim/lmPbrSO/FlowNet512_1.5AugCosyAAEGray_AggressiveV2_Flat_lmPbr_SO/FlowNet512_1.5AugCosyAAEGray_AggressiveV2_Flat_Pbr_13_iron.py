_base_ = "./FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_01_ape.py"
OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/iron"
DATASETS = dict(TRAIN=("lm_pbr_iron_train",), TEST=("lm_real_iron_test",))

# bbnc6
# objects  iron    Avg(1)
# ad_2     10.21   10.21
# ad_5     95.40   95.40
# ad_10    100.00  100.00
# rete_2   41.57   41.57
# rete_5   99.90   99.90
# rete_10  100.00  100.00
# re_2     41.68   41.68
# re_5     99.90   99.90
# re_10    100.00  100.00
# te_2     99.59   99.59
# te_5     100.00  100.00
# te_10    100.00  100.00
# proj_2   9.50    9.50
# proj_5   90.81   90.81
# proj_10  100.00  100.00
# re       2.23    2.23
# te       0.01    0.01
