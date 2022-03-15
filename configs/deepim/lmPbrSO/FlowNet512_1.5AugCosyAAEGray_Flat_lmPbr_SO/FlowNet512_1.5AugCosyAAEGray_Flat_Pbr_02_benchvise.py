_base_ = ["./FlowNet512_1.5AugCosyAAEGray_Flat_Pbr_01_ape.py"]


OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_Flat_lmPbr_SO/benchvise"

DATASETS = dict(TRAIN=("lm_pbr_benchvise_train",), TEST=("lm_real_benchvise_test",))

# bbnc7
# objects  benchvise  Avg(1)
# ad_2     9.12       9.12
# ad_5     44.52      44.52
# ad_10    90.69      90.69
# rete_2   45.97      45.97
# rete_5   99.71      99.71
# rete_10  100.00     100.00
# re_2     63.43      63.43
# re_5     99.71      99.71
# re_10    100.00     100.00
# te_2     77.40      77.40
# te_5     100.00     100.00
# te_10    100.00     100.00
# proj_2   75.75      75.75
# proj_5   99.22      99.22
# proj_10  100.00     100.00
# re       1.80       1.80
# te       0.01       0.01

# init by mlBCE
# objects  benchvise  Avg(1)
# ad_2     9.21       9.21
# ad_5     44.52      44.52
# ad_10    90.79      90.79
# rete_2   46.46      46.46
# rete_5   99.52      99.52
# rete_10  100.00     100.00
# re_2     64.31      64.31
# re_5     99.52      99.52
# re_10    100.00     100.00
# te_2     77.50      77.50
# te_5     100.00     100.00
# te_10    100.00     100.00
# proj_2   75.85      75.85
# proj_5   99.22      99.22
# proj_10  100.00     100.00
# re       1.80       1.80
# te       0.01       0.01
