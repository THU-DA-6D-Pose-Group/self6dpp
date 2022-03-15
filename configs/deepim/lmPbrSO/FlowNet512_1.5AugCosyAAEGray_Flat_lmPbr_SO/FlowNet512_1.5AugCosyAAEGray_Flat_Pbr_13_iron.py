_base_ = ["./FlowNet512_1.5AugCosyAAEGray_Flat_Pbr_01_ape.py"]


OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_Flat_lmPbr_SO/iron"

DATASETS = dict(TRAIN=("lm_pbr_iron_train",), TEST=("lm_real_iron_test",))

# gu
# objects  iron    Avg(1)
# ad_2     11.54   11.54
# ad_5     93.97   93.97
# ad_10    100.00  100.00
# rete_2   38.92   38.92
# rete_5   99.69   99.69
# rete_10  100.00  100.00
# re_2     39.12   39.12
# re_5     99.69   99.69
# re_10    100.00  100.00
# te_2     99.39   99.39
# te_5     100.00  100.00
# te_10    100.00  100.00
# proj_2   9.81    9.81
# proj_5   90.19   90.19
# proj_10  100.00  100.00
# re       2.30    2.30
# te       0.01    0.01

# init by mlBCE
# objects  iron    Avg(1)
# ad_2     11.13   11.13
# ad_5     93.97   93.97
# ad_10    100.00  100.00
# rete_2   39.73   39.73
# rete_5   99.69   99.69
# rete_10  100.00  100.00
# re_2     39.94   39.94
# re_5     99.69   99.69
# re_10    100.00  100.00
# te_2     99.39   99.39
# te_5     100.00  100.00
# te_10    100.00  100.00
# proj_2   9.60    9.60
# proj_5   90.19   90.19
# proj_10  100.00  100.00
# re       2.29    2.29
# te       0.01    0.01
