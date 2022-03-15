#!/usr/bin/env bash
set -ex

# 1
./core/gdrn_modeling/save_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_01_02MasterChefCan.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/01_02MasterChefCan/model_final_wo_optim-8624c7f1.pth \
  DATASETS.DET_FILES_TEST="datasets/BOP_DATASETS/ycbv/test/test_bboxes/yolov4x_640_test672_augCosyAAEGray_ranger_ycbv_pbr_ycbv_train_16e.json," \
  DATASETS.TEST="ycbv_002_master_chef_can_train_real_aligned_Kuw,"


# 2
./core/gdrn_modeling/save_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_02_03CrackerBox.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/02_03CrackerBox/model_final_wo_optim-72f9c1da.pth \
  DATASETS.DET_FILES_TEST="datasets/BOP_DATASETS/ycbv/test/test_bboxes/yolov4x_640_test672_augCosyAAEGray_ranger_ycbv_pbr_ycbv_train_16e.json," \
  DATASETS.TEST="ycbv_003_cracker_box_train_real_aligned_Kuw,"


# 3
./core/gdrn_modeling/save_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_03_04SugarBox.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/03_04SugarBox/model_final_wo_optim-bf2dc932.pth \
  DATASETS.DET_FILES_TEST="datasets/BOP_DATASETS/ycbv/test/test_bboxes/yolov4x_640_test672_augCosyAAEGray_ranger_ycbv_pbr_ycbv_train_16e.json," \
  DATASETS.TEST="ycbv_004_sugar_box_train_real_aligned_Kuw,"



# 4
./core/gdrn_modeling/save_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_04_05TomatoSoupCan.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/04_05TomatoSoupCan/model_final_wo_optim-b142bb56.pth \
  DATASETS.DET_FILES_TEST="datasets/BOP_DATASETS/ycbv/test/test_bboxes/yolov4x_640_test672_augCosyAAEGray_ranger_ycbv_pbr_ycbv_train_16e.json," \
  DATASETS.TEST="ycbv_005_tomato_soup_can_train_real_aligned_Kuw,"



# 5
./core/gdrn_modeling/save_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_05_06MustardBottle.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/05_06MustardBottle/model_final_wo_optim-86dde7e2.pth \
  DATASETS.DET_FILES_TEST="datasets/BOP_DATASETS/ycbv/test/test_bboxes/yolov4x_640_test672_augCosyAAEGray_ranger_ycbv_pbr_ycbv_train_16e.json," \
  DATASETS.TEST="ycbv_006_mustard_bottle_train_real_aligned_Kuw,"



# 6
./core/gdrn_modeling/save_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_06_07TunaFishCan.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/06_07TunaFishCan/model_final_wo_optim-0b376921.pth \
  DATASETS.DET_FILES_TEST="datasets/BOP_DATASETS/ycbv/test/test_bboxes/yolov4x_640_test672_augCosyAAEGray_ranger_ycbv_pbr_ycbv_train_16e.json," \
  DATASETS.TEST="ycbv_007_tuna_fish_can_train_real_aligned_Kuw,"


# 7
./core/gdrn_modeling/save_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_07_08PuddingBox.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/07_08PuddingBox/model_final_wo_optim-23fb01c0.pth \
  DATASETS.DET_FILES_TEST="datasets/BOP_DATASETS/ycbv/test/test_bboxes/yolov4x_640_test672_augCosyAAEGray_ranger_ycbv_pbr_ycbv_train_16e.json," \
  DATASETS.TEST="ycbv_008_pudding_box_train_real_aligned_Kuw,"


# 8
./core/gdrn_modeling/save_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_08_09GelatinBox.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/08_09GelatinBox/model_final_wo_optim-1256bdbd.pth \
  DATASETS.DET_FILES_TEST="datasets/BOP_DATASETS/ycbv/test/test_bboxes/yolov4x_640_test672_augCosyAAEGray_ranger_ycbv_pbr_ycbv_train_16e.json," \
  DATASETS.TEST="ycbv_009_gelatin_box_train_real_aligned_Kuw,"


# 9
./core/gdrn_modeling/save_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_09_10PottedMeatCan.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/09_10PottedMeatCan/model_final_wo_optim-cc1232e2.pth \
  DATASETS.DET_FILES_TEST="datasets/BOP_DATASETS/ycbv/test/test_bboxes/yolov4x_640_test672_augCosyAAEGray_ranger_ycbv_pbr_ycbv_train_16e.json," \
  DATASETS.TEST="ycbv_010_potted_meat_can_train_real_aligned_Kuw,"


# 10
./core/gdrn_modeling/save_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_10_11Banana.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/10_11Banana/model_final_wo_optim-427d6321.pth \
  DATASETS.DET_FILES_TEST="datasets/BOP_DATASETS/ycbv/test/test_bboxes/yolov4x_640_test672_augCosyAAEGray_ranger_ycbv_pbr_ycbv_train_16e.json," \
  DATASETS.TEST="ycbv_011_banana_train_real_aligned_Kuw,"


# 11
./core/gdrn_modeling/save_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_11_19PitcherBase.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/11_19PitcherBase/model_final_wo_optim-f052dafc.pth \
  DATASETS.DET_FILES_TEST="datasets/BOP_DATASETS/ycbv/test/test_bboxes/yolov4x_640_test672_augCosyAAEGray_ranger_ycbv_pbr_ycbv_train_16e.json," \
  DATASETS.TEST="ycbv_019_pitcher_base_train_real_aligned_Kuw,"


# 12
./core/gdrn_modeling/save_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_12_21BleachCleanser.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/12_21BleachCleanser/model_final_wo_optim-59a61f06.pth \
  DATASETS.DET_FILES_TEST="datasets/BOP_DATASETS/ycbv/test/test_bboxes/yolov4x_640_test672_augCosyAAEGray_ranger_ycbv_pbr_ycbv_train_16e.json," \
  DATASETS.TEST="ycbv_021_bleach_cleanser_train_real_aligned_Kuw,"


# 13
./core/gdrn_modeling/save_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_13_24Bowl.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/13_24Bowl/model_final_wo_optim-55b952fc.pth \
  DATASETS.DET_FILES_TEST="datasets/BOP_DATASETS/ycbv/test/test_bboxes/yolov4x_640_test672_augCosyAAEGray_ranger_ycbv_pbr_ycbv_train_16e.json," \
  DATASETS.TEST="ycbv_024_bowl_train_real_aligned_Kuw,"


# 14
./core/gdrn_modeling/save_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_14_25Mug.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/14_25Mug/model_final_wo_optim-6b280ec5.pth \
  DATASETS.DET_FILES_TEST="datasets/BOP_DATASETS/ycbv/test/test_bboxes/yolov4x_640_test672_augCosyAAEGray_ranger_ycbv_pbr_ycbv_train_16e.json," \
  DATASETS.TEST="ycbv_025_mug_train_real_aligned_Kuw,"


# 15
./core/gdrn_modeling/save_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_15_35PowerDrill.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/15_35PowerDrill/model_final_wo_optim-0769bee7.pth \
  DATASETS.DET_FILES_TEST="datasets/BOP_DATASETS/ycbv/test/test_bboxes/yolov4x_640_test672_augCosyAAEGray_ranger_ycbv_pbr_ycbv_train_16e.json," \
  DATASETS.TEST="ycbv_035_power_drill_train_real_aligned_Kuw,"


# 16
./core/gdrn_modeling/save_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_16_36WoodBlock.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/16_36WoodBlock/model_final_wo_optim-baaa72c3.pth \
  DATASETS.DET_FILES_TEST="datasets/BOP_DATASETS/ycbv/test/test_bboxes/yolov4x_640_test672_augCosyAAEGray_ranger_ycbv_pbr_ycbv_train_16e.json," \
  DATASETS.TEST="ycbv_036_wood_block_train_real_aligned_Kuw,"


# 17
./core/gdrn_modeling/save_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_17_37Scissors.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/17_37Scissors/model_final_wo_optim-eb042de2.pth \
  DATASETS.DET_FILES_TEST="datasets/BOP_DATASETS/ycbv/test/test_bboxes/yolov4x_640_test672_augCosyAAEGray_ranger_ycbv_pbr_ycbv_train_16e.json," \
  DATASETS.TEST="ycbv_037_scissors_train_real_aligned_Kuw,"



# 18
./core/gdrn_modeling/save_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_18_40LargeMarker.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/18_40LargeMarker/model_final_wo_optim-3c805088.pth \
  DATASETS.DET_FILES_TEST="datasets/BOP_DATASETS/ycbv/test/test_bboxes/yolov4x_640_test672_augCosyAAEGray_ranger_ycbv_pbr_ycbv_train_16e.json," \
  DATASETS.TEST="ycbv_040_large_marker_train_real_aligned_Kuw,"


# 19
./core/gdrn_modeling/save_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_19_51LargeClamp.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/19_51LargeClamp/model_final_wo_optim-9643daa1.pth \
  DATASETS.DET_FILES_TEST="datasets/BOP_DATASETS/ycbv/test/test_bboxes/yolov4x_640_test672_augCosyAAEGray_ranger_ycbv_pbr_ycbv_train_16e.json," \
  DATASETS.TEST="ycbv_051_large_clamp_train_real_aligned_Kuw,"


# 20
./core/gdrn_modeling/save_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_20_52ExtraLargeClamp.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/20_52ExtraLargeClamp/model_final_wo_optim-82f2dafa.pth \
  DATASETS.DET_FILES_TEST="datasets/BOP_DATASETS/ycbv/test/test_bboxes/yolov4x_640_test672_augCosyAAEGray_ranger_ycbv_pbr_ycbv_train_16e.json," \
  DATASETS.TEST="ycbv_052_extra_large_clamp_train_real_aligned_Kuw,"


# 21
./core/gdrn_modeling/save_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_21_61FoamBrick.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/21_61FoamBrick/model_final_wo_optim-375a4cba.pth \
  DATASETS.DET_FILES_TEST="datasets/BOP_DATASETS/ycbv/test/test_bboxes/yolov4x_640_test672_augCosyAAEGray_ranger_ycbv_pbr_ycbv_train_16e.json," \
  DATASETS.TEST="ycbv_061_foam_brick_train_real_aligned_Kuw,"
