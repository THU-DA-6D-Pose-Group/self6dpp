#!/usr/bin/env bash
set -ex

python core/gdrn_modeling/engine/test_utils.py \
    --result_dir output/gdrn/lmoPbrSO/resnest50d_online_AugCosyAAEGray_mlBCE_DoubleMask_lmo_pbr_100e/merged-bop-test-faster-rcnn/ \
    --result_names faster-rcnn-resnest50d-online-AugCosyAAEGray-mlBCE-DoubleMask-lmo-pbr-100e-merged-bop-test-test-iter0_lmo-test.csv \
    --dataset lmo \
    --split test \
    --split-type "" \
    --targets_name test_targets_bop19.json \
    --error_types mspd,mssd,vsd,reS,teS,reteS,ad \
    --render_type cpp


# objects           ape    can    cat    driller  duck   eggbox  glue   holepuncher  Avg(8)
# mspd_5:50         92.11  94.77  91.23  92.30    90.39  55.94   86.43  89.85        86.82
# mssd_0.050:0.500  70.57  88.24  68.77  90.00    57.39  27.17   66.07  69.10        67.79
# vsd_0.050:0.500   50.66  71.74  52.11  72.91    52.23  19.28   48.32  49.32        52.69
# reS_2             16.57  58.79  19.88  62.00    11.67  0.56    6.43   26.00        25.24
# reS_5             65.14  87.44  54.39  91.50    27.78  12.22   33.57  67.00        54.88
# reS_10            89.71  98.99  85.38  99.50    72.78  50.56   77.14  93.50        83.45
# teS_2             78.86  87.44  59.65  80.00    61.67  10.56   49.29  52.50        59.99
# teS_5             96.00  97.49  90.06  98.50    97.22  42.78   79.29  97.00        87.29
# teS_10            97.71  99.50  95.32  99.00    98.89  62.22   95.00  98.00        93.21
# reteS_2           15.43  51.76  16.96  50.00    8.89   0.00    4.29   17.00        20.54
# reteS_5           64.00  85.43  53.80  90.50    27.78  9.44    30.71  67.00        53.58
# reteS_10          89.14  98.99  84.80  98.50    72.22  48.89   76.43  93.50        82.81
# ad_0.020          1.71   7.54   1.17   7.50     0.00   0.56    10.71  0.00         3.65
# ad_0.050          9.14   50.75  19.30  50.50    1.11   9.44    50.71  7.50         24.81
# ad_0.100          44.57  86.93  47.95  89.50    12.78  35.00   72.86  35.00        53.07
