#!/usr/bin/env bash
set -ex
# 1 ape
./core/deepim/test_deepim.sh \
  configs/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_01_ape_lmo_bop_test.py 1 \
  output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/ape/model_final_wo_optim-c5ab39d4.pth


# 5 can
./core/deepim/test_deepim.sh \
  configs/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_05_can_lmo_bop_test.py 1 \
  output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/can/model_final_wo_optim-29275594.pth


# 6 cat
./core/deepim/test_deepim.sh \
  configs/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_06_cat_lmo_bop_test.py 1 \
  output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/cat/model_final_wo_optim-bd14a802.pth



# 8 driller
./core/deepim/test_deepim.sh \
  configs/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_08_driller_lmo_bop_test.py 1 \
  output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/driller/model_final_wo_optim-45c5dd36.pth



# 9 duck
./core/deepim/test_deepim.sh \
  configs/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_09_duck_lmo_bop_test.py 1 \
  output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/duck/model_final_wo_optim-e28d46ee.pth



# 10 eggbox
./core/deepim/test_deepim.sh \
  configs/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_10_eggbox_lmo_bop_test.py 1 \
  output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/eggbox/model_final_wo_optim-bb583866.pth



# 11 glue
./core/deepim/test_deepim.sh \
  configs/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_11_glue_lmo_bop_test.py 1 \
  output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/glue/model_final_wo_optim-8013fa70.pth



# 12 holepuncher
./core/deepim/test_deepim.sh \
  configs/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_12_holepuncher_lmo_bop_test.py 1 \
  output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/holepuncher/model_final_wo_optim-30e67e31.pth

# eval iter0
python core/gdrn_modeling/engine/test_utils.py \
    --result_dir output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/merged-bop-test-iter0/  \
    --result_names FlowNet512-1.5AugCosyAAEGray-AggressiveV2-Flat-Pbr-merged-lmo-bop-test-test-iter0_lmo-test.csv \
    --dataset lmo \
    --split test \
    --split-type "" \
    --targets_name test_targets_bop19.json \
    --error_types mspd,mssd,vsd,reS,teS,reteS,ad \
    --render_type cpp
#objects           ape    can    cat    driller  duck   eggbox  glue   holepuncher  Avg(8)
#mspd_5:50         92.46  94.02  91.23  91.80    90.72  51.61   87.50  89.10        86.19
#mssd_0.050:0.500  70.29  87.84  69.59  89.05    57.67  26.00   66.07  67.95        67.40
#vsd_0.050:0.500   50.68  71.76  52.78  70.66    52.17  18.63   48.14  48.51        52.25
#reS_2             14.86  59.30  19.30  59.00    11.11  0.56    5.00   27.00        24.51
#reS_5             64.57  86.43  53.80  90.00    27.22  14.44   32.86  65.50        54.35
#reS_10            89.14  98.49  85.38  98.50    72.22  47.78   73.57  93.50        82.32
#teS_2             78.29  85.43  63.74  74.00    61.67  12.22   50.00  52.50        59.73
#teS_5             94.86  99.50  88.89  98.00    97.22  38.89   78.57  96.50        86.55
#teS_10            97.14  99.50  95.32  99.00    98.89  58.33   95.71  97.00        92.61
#reteS_2           13.71  51.26  16.37  46.50    10.00  0.00    2.86   18.00        19.84
#reteS_5           62.86  85.93  53.80  89.00    27.22  10.56   31.43  65.50        53.29
#reteS_10          88.57  97.99  84.80  97.50    72.22  44.44   72.86  93.50        81.49
#ad_0.020          0.57   7.04   1.17   10.50    0.00   1.67    9.29   0.00         3.78
#ad_0.050          12.00  50.75  17.54  48.50    2.78   10.56   52.86  8.00         25.37
#ad_0.100          46.86  85.43  45.03  87.00    13.89  31.67   74.29  33.50        52.21

# eval iter4
python core/gdrn_modeling/engine/test_utils.py \
    --result_dir output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/merged-bop-test-iter4/  \
    --result_names FlowNet512-1.5AugCosyAAEGray-AggressiveV2-Flat-Pbr-merged-lmo-bop-test-test-iter4_lmo-test.csv \
    --dataset lmo \
    --split test \
    --split-type "" \
    --targets_name test_targets_bop19.json \
    --error_types mspd,mssd,vsd,reS,teS,reteS,ad \
    --render_type cpp
#objects           ape    can     cat    driller  duck   eggbox  glue   holepuncher  Avg(8)
#mspd_5:50         92.46  95.48   92.81  95.25    91.06  53.72   88.50  87.95        87.29
#mssd_0.050:0.500  73.71  94.22   73.80  93.80    62.94  37.39   75.21  64.15        72.28
#vsd_0.050:0.500   52.33  76.70   58.96  80.91    57.75  24.37   55.76  44.04        56.80
#reS_2             14.29  62.31   14.04  66.00    14.44  1.11    9.29   24.50        25.75
#reS_5             62.86  87.94   58.48  98.00    29.44  28.33   37.14  64.50        58.34
#reS_10            88.57  98.99   84.80  100.00   73.33  55.00   87.14  91.50        84.92
#teS_2             84.57  96.98   73.10  93.50    77.78  16.11   62.86  40.50        68.18
#teS_5             96.00  100.00  90.64  97.50    96.67  55.00   92.14  96.00        90.49
#teS_10            97.14  100.00  95.32  99.00    98.89  62.22   97.14  97.00        93.34
#reteS_2           12.00  60.80   12.87  64.00    13.89  0.00    7.14   12.50        22.90
#reteS_5           61.71  87.94   58.48  95.50    29.44  26.11   37.14  64.50        57.60
#reteS_10          88.57  98.99   84.80  99.00    72.78  53.33   87.14  91.50        84.51
#ad_0.020          0.00   17.09   2.92   24.00    0.56   2.22    14.29  1.00         7.76
#ad_0.050          14.86  78.89   25.15  81.00    5.56   15.00   63.57  5.00         36.13
#ad_0.100          53.71  96.98   60.23  96.50    27.78  51.11   89.29  24.50        62.51
