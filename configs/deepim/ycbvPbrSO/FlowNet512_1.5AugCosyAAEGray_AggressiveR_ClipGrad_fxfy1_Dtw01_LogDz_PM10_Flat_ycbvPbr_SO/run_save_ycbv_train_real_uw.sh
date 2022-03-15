#!/usr/bin/env bash
set -ex

# 01
./core/deepim/save_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_Pbr_01_02MasterChefCan.py 4 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/01_02MasterChefCan/model_final_wo_optim-2de2b4e3.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/ycbv/test/init_poses/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_ycbvTrainRealUw_GdrnPose_withYolov4PbrBbox.json," \
  DATASETS.TEST="ycbv_002_master_chef_can_train_real_uw,"

# 02
./core/deepim/save_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_Pbr_02_03CrackerBox.py 5 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/02_03CrackerBox/model_final_wo_optim-41082f8a.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/ycbv/test/init_poses/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_ycbvTrainRealUw_GdrnPose_withYolov4PbrBbox.json," \
  DATASETS.TEST="ycbv_003_cracker_box_train_real_uw,"


# 03
./core/deepim/save_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_Pbr_03_04SugarBox.py 6 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/03_04SugarBox/model_final_wo_optim-e09dec3e.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/ycbv/test/init_poses/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_ycbvTrainRealUw_GdrnPose_withYolov4PbrBbox.json," \
  DATASETS.TEST="ycbv_004_sugar_box_train_real_uw,"

# 4
./core/deepim/save_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_Pbr_04_05TomatoSoupCan.py 7 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/04_05TomatoSoupCan/model_final_wo_optim-5641f5d3.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/ycbv/test/init_poses/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_ycbvTrainRealUw_GdrnPose_withYolov4PbrBbox.json," \
  DATASETS.TEST="ycbv_005_tomato_soup_can_train_real_uw,"

# 5
./core/deepim/save_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_Pbr_05_06MustardBottle.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/05_06MustardBottle/model_final_wo_optim-6ce23e94.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/ycbv/test/init_poses/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_ycbvTrainRealUw_GdrnPose_withYolov4PbrBbox.json," \
  DATASETS.TEST="ycbv_006_mustard_bottle_train_real_uw,"

# 6
./core/deepim/save_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_Pbr_06_07TunaFishCan.py   1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/06_07TunaFishCan/model_final_wo_optim-0a768962.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/ycbv/test/init_poses/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_ycbvTrainRealUw_GdrnPose_withYolov4PbrBbox.json," \
  DATASETS.TEST="ycbv_007_tuna_fish_can_train_real_uw,"

# 7
./core/deepim/save_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_Pbr_07_08PuddingBox.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/07_08PuddingBox/model_final_wo_optim-f2f2cf73.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/ycbv/test/init_poses/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_ycbvTrainRealUw_GdrnPose_withYolov4PbrBbox.json," \
  DATASETS.TEST="ycbv_008_pudding_box_train_real_uw,"

# 8
./core/deepim/save_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_Pbr_08_09GelatinBox.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/08_09GelatinBox/model_final_wo_optim-a303aa1e.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/ycbv/test/init_poses/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_ycbvTrainRealUw_GdrnPose_withYolov4PbrBbox.json," \
  DATASETS.TEST="ycbv_009_gelatin_box_train_real_uw,"

# 9
./core/deepim/save_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_Pbr_09_10PottedMeatCan.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/09_10PottedMeatCan/model_final_wo_optim-84a56ffd.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/ycbv/test/init_poses/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_ycbvTrainRealUw_GdrnPose_withYolov4PbrBbox.json," \
  DATASETS.TEST="ycbv_010_potted_meat_can_train_real_uw,"

# 10
./core/deepim/save_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_Pbr_10_11Banana.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/10_11Banana/model_final_wo_optim-83947126.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/ycbv/test/init_poses/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_ycbvTrainRealUw_GdrnPose_withYolov4PbrBbox.json," \
  DATASETS.TEST="ycbv_011_banana_train_real_uw,"

# 11
./core/deepim/save_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_Pbr_11_19PitcherBase.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/11_19PitcherBase/model_final_wo_optim-af1c7e62.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/ycbv/test/init_poses/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_ycbvTrainRealUw_GdrnPose_withYolov4PbrBbox.json," \
  DATASETS.TEST="ycbv_019_pitcher_base_train_real_uw,"

# 12
./core/deepim/save_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_Pbr_12_21BleachCleanser.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/12_21BleachCleanser/model_final_wo_optim-5d740a46.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/ycbv/test/init_poses/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_ycbvTrainRealUw_GdrnPose_withYolov4PbrBbox.json," \
  DATASETS.TEST="ycbv_021_bleach_cleanser_train_real_uw,"

# 13
./core/deepim/save_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_Pbr_13_24Bowl.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/13_24Bowl/model_final_wo_optim-f11815d3.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/ycbv/test/init_poses/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_ycbvTrainRealUw_GdrnPose_withYolov4PbrBbox.json," \
  DATASETS.TEST="ycbv_024_bowl_train_real_uw,"

# 14
./core/deepim/save_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_Pbr_14_25Mug.py 2 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/14_25Mug/model_final_wo_optim-e4824065.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/ycbv/test/init_poses/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_ycbvTrainRealUw_GdrnPose_withYolov4PbrBbox.json," \
  DATASETS.TEST="ycbv_025_mug_train_real_uw,"

# 15
./core/deepim/save_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_Pbr_15_35PowerDrill.py 2 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/15_35PowerDrill/model_final_wo_optim-30d7d1da.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/ycbv/test/init_poses/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_ycbvTrainRealUw_GdrnPose_withYolov4PbrBbox.json," \
  DATASETS.TEST="ycbv_035_power_drill_train_real_uw,"

# 16
./core/deepim/save_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_Pbr_16_36WoodBlock.py 3 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/16_36WoodBlock/model_final_wo_optim-fbb38751.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/ycbv/test/init_poses/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_ycbvTrainRealUw_GdrnPose_withYolov4PbrBbox.json," \
  DATASETS.TEST="ycbv_036_wood_block_train_real_uw,"

# 17
./core/deepim/save_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_Pbr_17_37Scissors.py 4 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/17_37Scissors/model_final_wo_optim-5068c6bb.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/ycbv/test/init_poses/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_ycbvTrainRealUw_GdrnPose_withYolov4PbrBbox.json," \
  DATASETS.TEST="ycbv_037_scissors_train_real_uw,"

# 18
./core/deepim/save_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_Pbr_18_40LargeMarker.py 0 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/18_40LargeMarker/model_final_wo_optim-e8d5867c.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/ycbv/test/init_poses/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_ycbvTrainRealUw_GdrnPose_withYolov4PbrBbox.json," \
  DATASETS.TEST="ycbv_040_large_marker_train_real_uw,"

# 19
./core/deepim/save_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_Pbr_19_51LargeClamp.py 3 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/19_51LargeClamp/model_final_wo_optim-1ea79b34.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/ycbv/test/init_poses/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_ycbvTrainRealUw_GdrnPose_withYolov4PbrBbox.json," \
  DATASETS.TEST="ycbv_051_large_clamp_train_real_uw,"

# 20
./core/deepim/save_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_Pbr_20_52ExtraLargeClamp.py 3 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/20_52ExtraLargeClamp/model_final_wo_optim-cb595297.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/ycbv/test/init_poses/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_ycbvTrainRealUw_GdrnPose_withYolov4PbrBbox.json," \
  DATASETS.TEST="ycbv_052_extra_large_clamp_train_real_uw,"

# 21
./core/deepim/save_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_Pbr_21_61FoamBrick.py 0 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/21_61FoamBrick/model_final_wo_optim-d3757ca1.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/BOP_DATASETS/ycbv/test/init_poses/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_ycbvTrainRealUw_GdrnPose_withYolov4PbrBbox.json," \
  DATASETS.TEST="ycbv_061_foam_brick_train_real_uw,"



#python core/deepim/engine/test_utils.py \
#    --result_dir output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/02_03CrackerBox/inference_model_final_wo_optim-41082f8a/ycbv_test/ \
#    --result_names FlowNet512-1.5AugCosyAAEGray-AggressiveR-ClipGrad-fxfy1-Dtw01-LogDz-PM10-Flat-Pbr-02-03CrackerBox-test-iter4_ycbvposecnn-test.csv \
#    --dataset ycbvposecnn \
#    --split test \
#    --split-type "" \
#    --render_type cpp \
#    --targets_name ycbv_test_targets_keyframe.json \
#    --error_types AUCadd,AUCadi,AUCad,ABSadd,ABSadi,ABSad,reS,teS,reteS,ad


python core/deepim/engine/test_utils.py \
    --result_dir output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/merged-ycbv-test/ \
    --result_names FlowNet512-1.5AugCosyAAEGray-AggressiveR-ClipGrad-fxfy1-Dtw01-LogDz-PM10-Flat-Pbr-merged-test-iter4_ycbvposecnn-test.csv \
    --dataset ycbvposecnn \
    --split test \
    --split-type "" \
    --targets_name ycbv_test_targets_keyframe.json \
    --error_types AUCadd,AUCadi,AUCad,ABSadd,ABSadi,ABSad,reS,teS,reteS,ad \
    --render_type cpp
