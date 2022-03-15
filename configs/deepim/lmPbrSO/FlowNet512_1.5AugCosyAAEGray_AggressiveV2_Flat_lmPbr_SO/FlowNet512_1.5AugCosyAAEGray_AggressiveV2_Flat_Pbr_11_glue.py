_base_ = "./FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_01_ape.py"
OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/glue"
DATASETS = dict(TRAIN=("lm_pbr_glue_train",), TEST=("lm_real_glue_test",))

# bbnc6
# objects  glue   Avg(1)
# ad_2     21.14  21.14
# ad_5     66.99  66.99
# ad_10    93.34  93.34
# rete_2   7.43   7.43
# rete_5   77.12  77.12
# rete_10  98.55  98.55
# re_2     9.85   9.85
# re_5     79.25  79.25
# re_10    98.55  98.55
# te_2     67.37  67.37
# te_5     97.10  97.10
# te_10    99.90  99.90
# proj_2   18.92  18.92
# proj_5   92.95  92.95
# proj_10  99.90  99.90
# re       3.88   3.88
# te       0.02   0.02
