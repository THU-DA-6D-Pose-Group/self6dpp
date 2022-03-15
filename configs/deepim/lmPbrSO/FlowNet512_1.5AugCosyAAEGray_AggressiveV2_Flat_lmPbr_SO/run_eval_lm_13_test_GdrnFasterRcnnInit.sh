#!/usr/bin/env bash
set -ex

# 1 ape
./core/deepim/test_deepim.sh \
  configs/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_01_ape.py 1 \
  output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/ape/model_final_wo_optim-c5ab39d4.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/lm/test/init_poses/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e_so_wFasterRcnnPbrBboxGdrnInit_wDeepimPbrPose_lm_13_all.json," \
  DATASETS.TEST="lm_real_ape_test,"

# 2 benchvise
./core/deepim/test_deepim.sh \
  configs/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_02_benchvise.py 1 \
  output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/benchvise/model_final_wo_optim-956eb2eb.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/lm/test/init_poses/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e_so_wFasterRcnnPbrBboxGdrnInit_wDeepimPbrPose_lm_13_all.json," \
  DATASETS.TEST="lm_real_benchvise_test,"

# 4 camera
./core/deepim/test_deepim.sh \
  configs/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_04_camera.py 1 \
  output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/camera/model_final_wo_optim-520e2c6d.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/lm/test/init_poses/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e_so_wFasterRcnnPbrBboxGdrnInit_wDeepimPbrPose_lm_13_all.json," \
  DATASETS.TEST="lm_real_camera_test,"

# 5 can
./core/deepim/test_deepim.sh \
  configs/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_05_can.py 1 \
  output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/can/model_final_wo_optim-29275594.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/lm/test/init_poses/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e_so_wFasterRcnnPbrBboxGdrnInit_wDeepimPbrPose_lm_13_all.json," \
  DATASETS.TEST="lm_real_can_test,"

# 6 cat
./core/deepim/test_deepim.sh \
  configs/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_06_cat.py 1 \
  output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/cat/model_final_wo_optim-bd14a802.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/lm/test/init_poses/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e_so_wFasterRcnnPbrBboxGdrnInit_wDeepimPbrPose_lm_13_all.json," \
  DATASETS.TEST="lm_real_cat_test,"


# 8 driller
./core/deepim/test_deepim.sh \
  configs/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_08_driller.py 1 \
  output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/driller/model_final_wo_optim-45c5dd36.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/lm/test/init_poses/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e_so_wFasterRcnnPbrBboxGdrnInit_wDeepimPbrPose_lm_13_all.json," \
  DATASETS.TEST="lm_real_driller_test,"


# 9 duck
./core/deepim/test_deepim.sh \
  configs/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_09_duck.py 1 \
  output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/duck/model_final_wo_optim-e28d46ee.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/lm/test/init_poses/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e_so_wFasterRcnnPbrBboxGdrnInit_wDeepimPbrPose_lm_13_all.json," \
  DATASETS.TEST="lm_real_duck_test,"


# 10 eggbox
./core/deepim/test_deepim.sh \
  configs/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_10_eggbox.py 1 \
  output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/eggbox/model_final_wo_optim-bb583866.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/lm/test/init_poses/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e_so_wFasterRcnnPbrBboxGdrnInit_wDeepimPbrPose_lm_13_all.json," \
  DATASETS.TEST="lm_real_eggbox_test,"


# 11 glue
./core/deepim/test_deepim.sh \
  configs/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_11_glue.py 1 \
  output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/glue/model_final_wo_optim-8013fa70.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/lm/test/init_poses/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e_so_wFasterRcnnPbrBboxGdrnInit_wDeepimPbrPose_lm_13_all.json," \
  DATASETS.TEST="lm_real_glue_test,"


# 12 holepuncher
./core/deepim/test_deepim.sh \
  configs/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_12_holepuncher.py 1 \
  output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/holepuncher/model_final_wo_optim-30e67e31.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/lm/test/init_poses/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e_so_wFasterRcnnPbrBboxGdrnInit_wDeepimPbrPose_lm_13_all.json," \
  DATASETS.TEST="lm_real_holepuncher_test,"


# 13 iron
./core/deepim/test_deepim.sh \
  configs/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_13_iron.py 1 \
  output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/iron/model_final_wo_optim-485fd5c3.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/lm/test/init_poses/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e_so_wFasterRcnnPbrBboxGdrnInit_wDeepimPbrPose_lm_13_all.json," \
  DATASETS.TEST="lm_real_iron_test,"


# 14 lamp
./core/deepim/test_deepim.sh \
  configs/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_14_lamp.py 1 \
  output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/lamp/model_final_wo_optim-f169005c.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/lm/test/init_poses/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e_so_wFasterRcnnPbrBboxGdrnInit_wDeepimPbrPose_lm_13_all.json," \
  DATASETS.TEST="lm_real_lamp_test,"


# 15 phone
./core/deepim/test_deepim.sh \
  configs/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_15_phone.py 1 \
  output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/phone/model_final_wo_optim-61aad172.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/lm/test/init_poses/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e_so_wFasterRcnnPbrBboxGdrnInit_wDeepimPbrPose_lm_13_all.json," \
  DATASETS.TEST="lm_real_phone_test,"
