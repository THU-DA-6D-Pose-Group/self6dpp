#!/usr/bin/env bash
set -ex

# 1 ape
./core/deepim/save_deepim.sh \
  configs/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_01_ape_lmo_test.py 1 \
  output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/ape/model_final_wo_optim-c5ab39d4.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/lmo/test/init_poses/resnest50d_online_AugCosyAAEGray_mlBCE_DoubleMask_lmo_pbr_100e_so_NoBopTest_with_yolov4_pbr_bbox.json," \
  DATASETS.TEST="lmo_NoBopTest_train,"



# 5 can
./core/deepim/save_deepim.sh \
  configs/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_05_can_lmo_test.py 1 \
  output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/can/model_final_wo_optim-29275594.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/lmo/test/init_poses/resnest50d_online_AugCosyAAEGray_mlBCE_DoubleMask_lmo_pbr_100e_so_NoBopTest_with_yolov4_pbr_bbox.json," \
  DATASETS.TEST="lmo_NoBopTest_train,"


# 6 cat
./core/deepim/save_deepim.sh \
  configs/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_06_cat_lmo_test.py 1 \
  output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/cat/model_final_wo_optim-bd14a802.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/lmo/test/init_poses/resnest50d_online_AugCosyAAEGray_mlBCE_DoubleMask_lmo_pbr_100e_so_NoBopTest_with_yolov4_pbr_bbox.json," \
  DATASETS.TEST="lmo_NoBopTest_train,"



# 8 driller
./core/deepim/save_deepim.sh \
  configs/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_08_driller_lmo_test.py 1 \
  output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/driller/model_final_wo_optim-45c5dd36.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/lmo/test/init_poses/resnest50d_online_AugCosyAAEGray_mlBCE_DoubleMask_lmo_pbr_100e_so_NoBopTest_with_yolov4_pbr_bbox.json," \
  DATASETS.TEST="lmo_NoBopTest_train,"



# 9 duck
./core/deepim/save_deepim.sh \
  configs/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_09_duck_lmo_test.py 1 \
  output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/duck/model_final_wo_optim-e28d46ee.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/lmo/test/init_poses/resnest50d_online_AugCosyAAEGray_mlBCE_DoubleMask_lmo_pbr_100e_so_NoBopTest_with_yolov4_pbr_bbox.json," \
  DATASETS.TEST="lmo_NoBopTest_train,"



# 10 eggbox
./core/deepim/save_deepim.sh \
  configs/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_10_eggbox_lmo_test.py 1 \
  output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/eggbox/model_final_wo_optim-bb583866.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/lmo/test/init_poses/resnest50d_online_AugCosyAAEGray_mlBCE_DoubleMask_lmo_pbr_100e_so_NoBopTest_with_yolov4_pbr_bbox.json," \
  DATASETS.TEST="lmo_NoBopTest_train,"



# 11 glue
./core/deepim/save_deepim.sh \
  configs/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_11_glue_lmo_test.py 1 \
  output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/glue/model_final_wo_optim-8013fa70.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/lmo/test/init_poses/resnest50d_online_AugCosyAAEGray_mlBCE_DoubleMask_lmo_pbr_100e_so_NoBopTest_with_yolov4_pbr_bbox.json," \
  DATASETS.TEST="lmo_NoBopTest_train,"



# 12 holepuncher
./core/deepim/save_deepim.sh \
  configs/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_12_holepuncher_lmo_test.py 1 \
  output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/holepuncher/model_final_wo_optim-30e67e31.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/lmo/test/init_poses/resnest50d_online_AugCosyAAEGray_mlBCE_DoubleMask_lmo_pbr_100e_so_NoBopTest_with_yolov4_pbr_bbox.json," \
  DATASETS.TEST="lmo_NoBopTest_train,"

