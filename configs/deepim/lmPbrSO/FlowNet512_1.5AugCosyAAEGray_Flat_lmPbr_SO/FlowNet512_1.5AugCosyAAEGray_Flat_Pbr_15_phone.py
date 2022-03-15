_base_ = ["./FlowNet512_1.5AugCosyAAEGray_Flat_Pbr_01_ape.py"]


OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_Flat_lmPbr_SO/phone"

DATASETS = dict(TRAIN=("lm_pbr_phone_train",), TEST=("lm_real_phone_test",))

# rl3
# objects  phone   Avg(1)
# ad_2     7.18    7.18
# ad_5     68.93   68.93
# ad_10    95.09   95.09
# rete_2   36.73   36.73
# rete_5   97.17   97.17
# rete_10  100.00  100.00
# re_2     39.19   39.19
# re_5     97.17   97.17
# re_10    100.00  100.00
# te_2     94.33   94.33
# te_5     100.00  100.00
# te_10    100.00  100.00
# proj_2   23.80   23.80
# proj_5   97.83   97.83
# proj_10  100.00  100.00
# re       2.50    2.50
# te       0.01    0.01

# init by mlBCE
# objects  phone   Avg(1)
# ad_2     6.89    6.89
# ad_5     69.03   69.03
# ad_10    95.00   95.00
# rete_2   36.54   36.54
# rete_5   97.26   97.26
# rete_10  100.00  100.00
# re_2     38.90   38.90
# re_5     97.26   97.26
# re_10    100.00  100.00
# te_2     94.24   94.24
# te_5     100.00  100.00
# te_10    100.00  100.00
# proj_2   24.17   24.17
# proj_5   97.92   97.92
# proj_10  100.00  100.00
# re       2.50    2.50
# te       0.01    0.01
