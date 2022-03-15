#!/usr/bin/env bash
set -ex

# 1
./core/deepim/test_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_Pbr_01_02MasterChefCan_bop_test.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_ycbvPbr_SO/01_02MasterChefCan/model_final_wo_optim-f9b45add.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 2
./core/deepim/test_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_Pbr_02_03CrackerBox_bop_test.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_ycbvPbr_SO/02_03CrackerBox/model_final_wo_optim-8085b93a.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 3
./core/deepim/test_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_Pbr_03_04SugarBox_bop_test.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_ycbvPbr_SO/03_04SugarBox/model_final_wo_optim-09d8712e.pth \
  VAL.SAVE_BOP_CSV_ONLY=True


# 4
./core/deepim/test_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_Pbr_04_05TomatoSoupCan_bop_test.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_ycbvPbr_SO/04_05TomatoSoupCan/model_final_wo_optim-1b91bfde.pth \
  VAL.SAVE_BOP_CSV_ONLY=True


# 5
./core/deepim/test_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_Pbr_05_06MustardBottle_bop_test.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_ycbvPbr_SO/05_06MustardBottle/model_final_wo_optim-76d617ad.pth \
  VAL.SAVE_BOP_CSV_ONLY=True


# 6
./core/deepim/test_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_Pbr_06_07TunaFishCan_bop_test.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_ycbvPbr_SO/06_07TunaFishCan/model_final_wo_optim-df0d3e00.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 7
./core/deepim/test_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_Pbr_07_08PuddingBox_bop_test.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_ycbvPbr_SO/07_08PuddingBox/model_final_wo_optim-be3b4685.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 8
./core/deepim/test_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_Pbr_08_09GelatinBox_bop_test.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_ycbvPbr_SO/08_09GelatinBox/model_final_wo_optim-9cb36af3.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 9
./core/deepim/test_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_Pbr_09_10PottedMeatCan_bop_test.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_ycbvPbr_SO/09_10PottedMeatCan/model_final_wo_optim-0d651cea.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 10
./core/deepim/test_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_Pbr_10_11Banana_bop_test.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_ycbvPbr_SO/10_11Banana/model_final_wo_optim-2885507c.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 11
./core/deepim/test_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_Pbr_11_19PitcherBase_bop_test.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_ycbvPbr_SO/11_19PitcherBase/model_final_wo_optim-a6dbc5e6.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 12
./core/deepim/test_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_Pbr_12_21BleachCleanser_bop_test.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_ycbvPbr_SO/12_21BleachCleanser/model_final_wo_optim-94c0fbf0.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 13
./core/deepim/test_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_Pbr_13_24Bowl_bop_test.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_ycbvPbr_SO/13_24Bowl/model_final_wo_optim-d2cbe903.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 14
./core/deepim/test_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_Pbr_14_25Mug_bop_test.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_ycbvPbr_SO/14_25Mug/model_final_wo_optim-450d3409.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 15
./core/deepim/test_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_Pbr_15_35PowerDrill_bop_test.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_ycbvPbr_SO/15_35PowerDrill/model_final_wo_optim-ca66ffb9.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 16
./core/deepim/test_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_Pbr_16_36WoodBlock_bop_test.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_ycbvPbr_SO/16_36WoodBlock/model_final_wo_optim-3159154a.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 17
./core/deepim/test_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_Pbr_17_37Scissors_bop_test.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_ycbvPbr_SO/17_37Scissors/model_final_wo_optim-ae581f60.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 18
./core/deepim/test_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_Pbr_18_40LargeMarker_bop_test.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_ycbvPbr_SO/18_40LargeMarker/model_final_wo_optim-96fdb54f.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 19
./core/deepim/test_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_Pbr_19_51LargeClamp_bop_test.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_ycbvPbr_SO/19_51LargeClamp/model_final_wo_optim-081a3e92.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 20
./core/deepim/test_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_Pbr_20_52ExtraLargeClamp_bop_test.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_ycbvPbr_SO/20_52ExtraLargeClamp/model_final_wo_optim-22a5a504.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 21
./core/deepim/test_deepim.sh \
  configs/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_ycbvPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_Pbr_21_61FoamBrick_bop_test.py 1 \
  output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_ycbvPbr_SO/21_61FoamBrick/model_final_wo_optim-90d8317c.pth \
  VAL.SAVE_BOP_CSV_ONLY=True


# eval merged csv
#python core/deepim/engine/test_utils.py \
#    --result_dir output/deepim/ycbvPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV3_Flat_ycbvPbr_SO/merged-bop-iter4/ \
#    --result_names  FlowNet512-1.5AugCosyAAEGray-AggressiveV3-Flat-Pbr-merged-bop-test-test-iter4_ycbv-test.csv \
#    --dataset ycbv \
#    --split test \
#    --split-type "" \
#    --targets_name test_targets_bop19.json \
#    --error_types mspd,mssd,vsd,reS,teS,reteS,ad \
#    --render_type cpp
