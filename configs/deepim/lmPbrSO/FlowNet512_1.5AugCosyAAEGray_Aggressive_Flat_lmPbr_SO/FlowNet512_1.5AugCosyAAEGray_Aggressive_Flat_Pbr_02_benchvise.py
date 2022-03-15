_base_ = "./FlowNet512_1.5AugCosyAAEGray_Aggressive_Flat_Pbr_01_ape.py"
OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_Aggressive_Flat_lmPbr_SO/benchvise"
DATASETS = dict(TRAIN=("lm_pbr_benchvise_train",), TEST=("lm_real_benchvise_test",))

# bbnc7
# objects  benchvise  Avg(1)
# ad_2     10.77      10.77
# ad_5     48.01      48.01
# ad_10    92.92      92.92
# rete_2   43.55      43.55
# rete_5   99.32      99.32
# rete_10  100.00     100.00
# re_2     57.32      57.32
# re_5     99.32      99.32
# re_10    100.00     100.00
# te_2     82.25      82.25
# te_5     100.00     100.00
# te_10    100.00     100.00
# proj_2   74.88      74.88
# proj_5   99.22      99.22
# proj_10  100.00     100.00
# re       1.97       1.97
# te       0.01       0.01
