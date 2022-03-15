#!/usr/bin/env bash
set -ex

# 1 ape
./core/deepim/save_deepim.sh \
  configs/deepim/lmCropBlenderSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmCropBlender_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Blender_01_ape.py 1 \
  output/deepim/lmCropBlenderSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmCropBlender_SO/ape/model_final_wo_optim-8ab2290c.pth

# 2 benchvise
./core/deepim/save_deepim.sh \
  configs/deepim/lmCropBlenderSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmCropBlender_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Blender_02_benchvise.py 1 \
  output/deepim/lmCropBlenderSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmCropBlender_SO/benchvise/model_final_wo_optim-1a8226bf.pth

# 4 camera
./core/deepim/save_deepim.sh \
  configs/deepim/lmCropBlenderSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmCropBlender_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Blender_04_camera.py 1 \
  output/deepim/lmCropBlenderSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmCropBlender_SO/camera/model_final_wo_optim-99b5e014.pth

# 5 can
./core/deepim/save_deepim.sh \
  configs/deepim/lmCropBlenderSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmCropBlender_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Blender_05_can.py 1 \
  output/deepim/lmCropBlenderSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmCropBlender_SO/can/model_final_wo_optim-2ba25c4e.pth

# 6 cat
./core/deepim/save_deepim.sh \
  configs/deepim/lmCropBlenderSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmCropBlender_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Blender_06_cat.py 1 \
  output/deepim/lmCropBlenderSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmCropBlender_SO/cat/model_final_wo_optim-e3b25aaa.pth


# 8 driller
./core/deepim/save_deepim.sh \
  configs/deepim/lmCropBlenderSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmCropBlender_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Blender_08_driller.py 1 \
  output/deepim/lmCropBlenderSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmCropBlender_SO/driller/model_final_wo_optim-1bcb3533.pth


# 9 duck
./core/deepim/save_deepim.sh \
  configs/deepim/lmCropBlenderSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmCropBlender_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Blender_09_duck.py 1 \
  output/deepim/lmCropBlenderSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmCropBlender_SO/duck/model_final_wo_optim-48ec2792.pth



# 12 holepuncher
./core/deepim/save_deepim.sh \
  configs/deepim/lmCropBlenderSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmCropBlender_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Blender_12_holepuncher.py 1 \
  output/deepim/lmCropBlenderSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmCropBlender_SO/holepuncher/model_final_wo_optim-deb2c268.pth


# 13 iron
./core/deepim/save_deepim.sh \
  configs/deepim/lmCropBlenderSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmCropBlender_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Blender_13_iron.py 1 \
  output/deepim/lmCropBlenderSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmCropBlender_SO/iron/model_final_wo_optim-94917683.pth


# 14 lamp
./core/deepim/save_deepim.sh \
  configs/deepim/lmCropBlenderSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmCropBlender_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Blender_14_lamp.py 1 \
  output/deepim/lmCropBlenderSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmCropBlender_SO/lamp/model_final_wo_optim-3b31c1da.pth


# 15 phone
./core/deepim/save_deepim.sh \
  configs/deepim/lmCropBlenderSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmCropBlender_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Blender_15_phone.py 1 \
  output/deepim/lmCropBlenderSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmCropBlender_SO/phone/model_final_wo_optim-96734341.pth





