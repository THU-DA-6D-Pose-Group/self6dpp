#!/usr/bin/env bash
set -ex

# 1 ape
./core/deepim/test_deepim.sh \
  configs/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_01_ape_lmo_bop_test.py 1 \
  output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/ape/model_final_wo_optim-c5ab39d4.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/lmo/test/init_poses/resnest50d_online_AugCosyAAEGray_mlBCE_DoubleMask_lmo_pbr_100e_so_lmo_test_Poses_wFasterRcnn101PbrBbox.json,"


# 5 can
./core/deepim/test_deepim.sh \
  configs/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_05_can_lmo_bop_test.py 1 \
  output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/can/model_final_wo_optim-29275594.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/lmo/test/init_poses/resnest50d_online_AugCosyAAEGray_mlBCE_DoubleMask_lmo_pbr_100e_so_lmo_test_Poses_wFasterRcnn101PbrBbox.json,"


# 6 cat
./core/deepim/test_deepim.sh \
  configs/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_06_cat_lmo_bop_test.py 1 \
  output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/cat/model_final_wo_optim-bd14a802.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/lmo/test/init_poses/resnest50d_online_AugCosyAAEGray_mlBCE_DoubleMask_lmo_pbr_100e_so_lmo_test_Poses_wFasterRcnn101PbrBbox.json,"



# 8 driller
./core/deepim/test_deepim.sh \
  configs/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_08_driller_lmo_bop_test.py 1 \
  output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/driller/model_final_wo_optim-45c5dd36.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/lmo/test/init_poses/resnest50d_online_AugCosyAAEGray_mlBCE_DoubleMask_lmo_pbr_100e_so_lmo_test_Poses_wFasterRcnn101PbrBbox.json,"



# 9 duck
./core/deepim/test_deepim.sh \
  configs/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_09_duck_lmo_bop_test.py 1 \
  output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/duck/model_final_wo_optim-e28d46ee.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/lmo/test/init_poses/resnest50d_online_AugCosyAAEGray_mlBCE_DoubleMask_lmo_pbr_100e_so_lmo_test_Poses_wFasterRcnn101PbrBbox.json,"



# 10 eggbox
./core/deepim/test_deepim.sh \
  configs/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_10_eggbox_lmo_bop_test.py 1 \
  output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/eggbox/model_final_wo_optim-bb583866.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/lmo/test/init_poses/resnest50d_online_AugCosyAAEGray_mlBCE_DoubleMask_lmo_pbr_100e_so_lmo_test_Poses_wFasterRcnn101PbrBbox.json,"



# 11 glue
./core/deepim/test_deepim.sh \
  configs/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_11_glue_lmo_bop_test.py 1 \
  output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/glue/model_final_wo_optim-8013fa70.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/lmo/test/init_poses/resnest50d_online_AugCosyAAEGray_mlBCE_DoubleMask_lmo_pbr_100e_so_lmo_test_Poses_wFasterRcnn101PbrBbox.json,"



# 12 holepuncher
./core/deepim/test_deepim.sh \
  configs/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_12_holepuncher_lmo_bop_test.py 1 \
  output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/holepuncher/model_final_wo_optim-30e67e31.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/lmo/test/init_poses/resnest50d_online_AugCosyAAEGray_mlBCE_DoubleMask_lmo_pbr_100e_so_lmo_test_Poses_wFasterRcnn101PbrBbox.json,"

# # eval iter0
# python core/gdrn_modeling/engine/test_utils.py \
#     --result_dir output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/merged-bop-test-iter0/  \
#     --result_names FlowNet512-1.5AugCosyAAEGray-AggressiveV2-Flat-Pbr-merged-lmo-bop-test-test-iter0_lmo-test.csv \
#     --dataset lmo \
#     --split test \
#     --split-type "" \
#     --targets_name test_targets_bop19.json \
#     --error_types mspd,mssd,vsd,reS,teS,reteS,ad \
#     --render_type cpp


# # eval iter4
# python core/gdrn_modeling/engine/test_utils.py \
#     --result_dir output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/merged_bop_test_iter4_GdrnFasterRcnn/  \
#     --result_names FlowNet512-1.5AugCosyAAEGray-AggressiveV2-Flat-Pbr-merged-lmo-bop-test-test-iter4_lmo-test.csv \
#     --dataset lmo \
#     --split test \
#     --split-type "" \
#     --targets_name test_targets_bop19.json \
#     --error_types mspd,mssd,vsd,reS,teS,reteS,ad \
#     --render_type cpp

# objects           ape    can    cat    driller  duck   eggbox  glue   holepuncher  Avg(8)
# mspd_5:50         92.29  95.93  92.57  95.25    90.44  58.78   87.93  88.85        87.93
# mssd_0.050:0.500  74.23  94.57  72.69  94.05    62.89  39.56   75.14  65.00        72.67
# vsd_0.050:0.500   52.64  76.47  57.85  80.85    57.60  26.34   55.98  45.31        57.09
# reS_2             14.86  63.32  14.04  66.50    13.89  1.67    9.29   24.50        26.01
# reS_5             62.86  87.44  60.82  98.00    30.00  30.56   37.86  65.00        59.07
# reS_10            90.29  99.50  84.80  100.00   72.78  60.56   87.14  93.50        86.07
# teS_2             85.14  97.49  69.01  93.50    78.33  15.56   62.14  42.50        67.96
# teS_5             96.57  99.50  91.81  98.00    97.22  61.11   91.43  96.50        91.52
# teS_10            97.71  99.50  95.32  99.00    98.89  65.56   95.71  98.00        93.71
# reteS_2           12.57  61.81  12.87  64.00    13.33  0.00    6.43   12.50        22.94
# reteS_5           61.71  87.44  60.82  96.00    30.00  28.89   37.86  65.00        58.46
# reteS_10          89.71  99.50  84.80  99.00    72.22  60.00   87.14  93.50        85.73
# ad_0.020          0.00   19.60  2.34   23.50    0.00   1.11    15.00  0.50         7.76
# ad_0.050          15.43  80.40  24.56  78.50    5.00   13.89   63.57  5.50         35.86
# ad_0.100          52.57  97.49  56.73  97.50    27.22  56.11   88.57  23.50        62.46
