_base_ = ["./FlowNet512_1.5AugAAE_Flat_Pbr_01_ape.py"]


OUTPUT_DIR = "output/deepim/lmPbrSO/FlowNet512_1.5AugAAE_Flat_lmPbr_SO/driller"

DATASETS = dict(TRAIN=("lm_pbr_driller_train",), TEST=("lm_real_driller_test",))

# bbnc10
# objects  driller  Avg(1)                                                                                                                                                                                    │···············································································································
# ad_2     50.45    50.45                                                                                                                                                                                     │···············································································································
# ad_5     92.07    92.07                                                                                                                                                                                     │···············································································································
# ad_10    100.00   100.00                                                                                                                                                                                    │···············································································································
# rete_2   84.14    84.14                                                                                                                                                                                     │···············································································································
# rete_5   99.90    99.90                                                                                                                                                                                     │···············································································································
# rete_10  100.00   100.00                                                                                                                                                                                    │···············································································································
# re_2     84.74    84.74                                                                                                                                                                                     │···············································································································
# re_5     99.90    99.90                                                                                                                                                                                     │···············································································································
# re_10    100.00   100.00                                                                                                                                                                                    │···············································································································
# te_2     99.31    99.31                                                                                                                                                                                     │···············································································································
# te_5     100.00   100.00                                                                                                                                                                                    │···············································································································
# te_10    100.00   100.00                                                                                                                                                                                    │···············································································································
# proj_2   85.13    85.13                                                                                                                                                                                     │···············································································································
# proj_5   99.01    99.01                                                                                                                                                                                     │···············································································································
# proj_10  100.00   100.00                                                                                                                                                                                    │···············································································································
# re       1.32     1.32                                                                                                                                                                                      │···············································································································
# te       0.01     0.01