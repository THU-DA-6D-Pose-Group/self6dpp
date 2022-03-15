#!/usr/bin/env bash
set -ex

# ape
./core/self6dpp/test_self6dpp.sh \
  configs/self6dpp/ssLMO/ss_mlBCE_MaskFull_PredFull_lr1e_5_lower_woCenter_refinePM10_lmoNoBopTest/ss_mlBCE_MaskFull_PredFull_lr1e_5_lower_woCenter_refinePM10_01_ape.py 1 \
  output/self6dpp/ssLMO/ss_mlBCE_MaskFull_PredFull_lr1e_5_lower_woCenter_refinePM10_lmoNoBopTest/ape/model_final_wo_optim-57c901fc.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# can
./core/self6dpp/test_self6dpp.sh \
  configs/self6dpp/ssLMO/ss_mlBCE_MaskFull_PredFull_lr1e_5_lower_woCenter_refinePM10_lmoNoBopTest/ss_mlBCE_MaskFull_PredFull_lr1e_5_lower_woCenter_refinePM10_05_can.py 1 \
  output/self6dpp/ssLMO/ss_mlBCE_MaskFull_PredFull_lr1e_5_lower_woCenter_refinePM10_lmoNoBopTest/can/model_final_wo_optim-db96d3dc.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# cat
./core/self6dpp/test_self6dpp.sh \
  configs/self6dpp/ssLMO/ss_mlBCE_MaskFull_PredFull_lr1e_5_lower_woCenter_refinePM10_lmoNoBopTest/ss_mlBCE_MaskFull_PredFull_lr1e_5_lower_woCenter_refinePM10_06_cat.py 1 \
  output/self6dpp/ssLMO/ss_mlBCE_MaskFull_PredFull_lr1e_5_lower_woCenter_refinePM10_lmoNoBopTest/cat/model_final_wo_optim-d27458fb.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# driller
./core/self6dpp/test_self6dpp.sh \
  configs/self6dpp/ssLMO/ss_mlBCE_MaskFull_PredFull_lr1e_5_lower_woCenter_refinePM10_lmoNoBopTest/ss_mlBCE_MaskFull_PredFull_lr1e_5_lower_woCenter_refinePM10_08_driller.py 1 \
  output/self6dpp/ssLMO/ss_mlBCE_MaskFull_PredFull_lr1e_5_lower_woCenter_refinePM10_lmoNoBopTest/driller/model_final_wo_optim-64eec6b2.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# duck
./core/self6dpp/test_self6dpp.sh \
  configs/self6dpp/ssLMO/ss_mlBCE_MaskFull_PredFull_lr1e_5_lower_woCenter_refinePM10_lmoNoBopTest/ss_mlBCE_MaskFull_PredFull_lr1e_5_lower_woCenter_refinePM10_09_duck.py 1 \
  output/self6dpp/ssLMO/ss_mlBCE_MaskFull_PredFull_lr1e_5_lower_woCenter_refinePM10_lmoNoBopTest/duck/model_final_wo_optim-5c6dc578.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# eggbox
./core/self6dpp/test_self6dpp.sh \
  configs/self6dpp/ssLMO/ss_mlBCE_MaskFull_PredFull_lr1e_5_lower_woCenter_refinePM10_lmoNoBopTest/ss_mlBCE_MaskFull_PredFull_lr1e_5_lower_woCenter_refinePM10_10_eggbox.py 1 \
  output/self6dpp/ssLMO/ss_mlBCE_MaskFull_PredFull_lr1e_5_lower_woCenter_refinePM10_lmoNoBopTest/eggbox/model_final_wo_optim-45db2b71.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# glue
./core/self6dpp/test_self6dpp.sh \
  configs/self6dpp/ssLMO/ss_mlBCE_MaskFull_PredFull_lr1e_5_lower_woCenter_refinePM10_lmoNoBopTest/ss_mlBCE_MaskFull_PredFull_lr1e_5_lower_woCenter_refinePM10_11_glue.py 1 \
  output/self6dpp/ssLMO/ss_mlBCE_MaskFull_PredFull_lr1e_5_lower_woCenter_refinePM10_lmoNoBopTest/glue/model_final_wo_optim-60598376.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# holepuncher
./core/self6dpp/test_self6dpp.sh \
  configs/self6dpp/ssLMO/ss_mlBCE_MaskFull_PredFull_lr1e_5_lower_woCenter_refinePM10_lmoNoBopTest/ss_mlBCE_MaskFull_PredFull_lr1e_5_lower_woCenter_refinePM10_12_holepuncher.py 1 \
  output/self6dpp/ssLMO/ss_mlBCE_MaskFull_PredFull_lr1e_5_lower_woCenter_refinePM10_lmoNoBopTest/holepuncher/model_final_wo_optim-a8606013.pth \
  VAL.SAVE_BOP_CSV_ONLY=True



#python core/self6dpp/engine/test_utils.py \
#    --result_dir output/self6dpp/ssLMO/ss_mlBCE_MaskFull_PredFull_lr1e_5_lower_woCenter_refinePM10_lmoNoBopTest/merged-bop-test/ \
#    --result_names ss-mlBCE-MaskFull-PredFull-lr1e-5-lower-woCenter-refinePM10-merged-bop-test-iter0_lmo-test.csv \
#    --dataset lmo \
#    --split test \
#    --split-type "" \
#    --targets_name test_targets_bop19.json \
#    --error_types mspd,mssd,vsd,reS,teS,reteS,ad \
#    --render_type cpp


#objects           ape    can     cat    driller  duck    eggbox  glue   holepuncher  Avg(8)
#mspd_5:50         92.80  94.62   91.64  91.85    91.00   54.39   86.64  88.45        86.57
#mssd_0.050:0.500  75.03  91.31   71.05  91.15    64.78   37.44   73.71  70.20        72.28
#vsd_0.050:0.500   53.24  73.33   54.91  73.35    58.91   25.43   53.21  49.11        55.66
#reS_2             16.00  59.30   8.77   45.50    12.22   1.11    2.86   20.00        20.72
#reS_5             65.71  88.94   56.14  85.50    28.89   17.78   30.00  62.00        54.37
#reS_10            89.71  98.99   84.80  99.00    71.11   53.89   78.57  92.00        83.51
#teS_2             84.57  95.98   69.01  87.50    80.56   21.67   60.00  56.50        69.47
#teS_5             97.14  99.50   88.89  98.50    97.78   53.33   93.57  97.00        90.71
#teS_10            97.71  100.00  94.74  100.00   100.00  61.67   95.71  97.00        93.35
#reteS_2           16.00  57.29   8.19   41.00    11.67   0.00    1.43   12.00        18.45
#reteS_5           65.71  88.44   56.14  84.00    28.89   16.11   30.00  62.00        53.91
#reteS_10          89.71  98.99   84.80  99.00    71.11   51.11   78.57  92.00        83.16
#ad_0.020          1.71   7.04    2.34   11.50    0.56    0.56    14.29  0.00         4.75
#ad_0.050          17.71  64.32   23.39  63.50    7.22    17.78   57.86  8.50         32.54
#ad_0.100          59.43  96.48   60.82  92.00    30.56   51.11   88.57  38.50        64.68