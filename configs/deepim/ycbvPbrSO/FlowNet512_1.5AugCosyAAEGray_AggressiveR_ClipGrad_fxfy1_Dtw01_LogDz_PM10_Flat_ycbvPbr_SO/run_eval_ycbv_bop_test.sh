#!/usr/bin/env bash
set -ex

# 01
./core/deepim/test_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_Pbr_01_02MasterChefCan_bop_test.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/01_02MasterChefCan/model_final_wo_optim-2de2b4e3.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 2
./core/deepim/test_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_Pbr_02_03CrackerBox_bop_test.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/02_03CrackerBox/model_final_wo_optim-41082f8a.pth \
  VAL.SAVE_BOP_CSV_ONLY=True


# 3
./core/deepim/test_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_Pbr_03_04SugarBox_bop_test.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/03_04SugarBox/model_final_wo_optim-e09dec3e.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 4
./core/deepim/test_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_Pbr_04_05TomatoSoupCan_bop_test.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/04_05TomatoSoupCan/model_final_wo_optim-5641f5d3.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 5
./core/deepim/test_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_Pbr_05_06MustardBottle_bop_test.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/05_06MustardBottle/model_final_wo_optim-6ce23e94.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 6
./core/deepim/test_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_Pbr_06_07TunaFishCan_bop_test.py   1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/06_07TunaFishCan/model_final_wo_optim-0a768962.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 7
./core/deepim/test_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_Pbr_07_08PuddingBox_bop_test.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/07_08PuddingBox/model_final_wo_optim-f2f2cf73.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 8
./core/deepim/test_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_Pbr_08_09GelatinBox_bop_test.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/08_09GelatinBox/model_final_wo_optim-a303aa1e.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 9
./core/deepim/test_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_Pbr_09_10PottedMeatCan_bop_test.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/09_10PottedMeatCan/model_final_wo_optim-84a56ffd.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 10
./core/deepim/test_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_Pbr_10_11Banana_bop_test.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/10_11Banana/model_final_wo_optim-83947126.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 11
./core/deepim/test_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_Pbr_11_19PitcherBase_bop_test.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/11_19PitcherBase/model_final_wo_optim-af1c7e62.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 12
./core/deepim/test_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_Pbr_12_21BleachCleanser_bop_test.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/12_21BleachCleanser/model_final_wo_optim-5d740a46.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 13
./core/deepim/test_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_Pbr_13_24Bowl_bop_test.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/13_24Bowl/model_final_wo_optim-f11815d3.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 14
./core/deepim/test_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_Pbr_14_25Mug_bop_test.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/14_25Mug/model_final_wo_optim-e4824065.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 15
./core/deepim/test_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_Pbr_15_35PowerDrill_bop_test.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/15_35PowerDrill/model_final_wo_optim-30d7d1da.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 16
./core/deepim/test_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_Pbr_16_36WoodBlock_bop_test.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/16_36WoodBlock/model_final_wo_optim-fbb38751.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 17
./core/deepim/test_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_Pbr_17_37Scissors_bop_test.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/17_37Scissors/model_final_wo_optim-5068c6bb.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 18
./core/deepim/test_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_Pbr_18_40LargeMarker_bop_test.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/18_40LargeMarker/model_final_wo_optim-e8d5867c.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 19
./core/deepim/test_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_Pbr_19_51LargeClamp_bop_test.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/19_51LargeClamp/model_final_wo_optim-1ea79b34.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 20
./core/deepim/test_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_Pbr_20_52ExtraLargeClamp_bop_test.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/20_52ExtraLargeClamp/model_final_wo_optim-cb595297.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 21
./core/deepim/test_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_Pbr_21_61FoamBrick_bop_test.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/21_61FoamBrick/model_final_wo_optim-d3757ca1.pth \
  VAL.SAVE_BOP_CSV_ONLY=True



#python core/deepim/engine/test_utils.py \
#    --result_dir output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveR_ClipGrad_fxfy1_Dtw01_LogDz_PM10_Flat_ycbvPbr_SO/merged-ycbv-test/ \
#    --result_names FlowNet512-1.5AugCosyAAEGray-AggressiveR-ClipGrad-fxfy1-Dtw01-LogDz-PM10-Flat-Pbr-merged-test-iter4_ycbvposecnn-test.csv \
#    --dataset ycbvposecnn \
#    --split test \
#    --split-type "" \
#    --targets_name ycbv_test_targets_keyframe.json \
#    --error_types AUCadd,AUCadi,AUCad,ABSadd,ABSadi,ABSad,reS,teS,reteS,ad \
#    --render_type cpp
