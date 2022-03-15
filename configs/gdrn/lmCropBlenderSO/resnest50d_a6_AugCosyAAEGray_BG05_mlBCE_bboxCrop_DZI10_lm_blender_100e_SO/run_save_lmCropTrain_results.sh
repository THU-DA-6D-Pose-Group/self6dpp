#!/usr/bin/env bash
set -ex

# 1 ape
./core/gdrn_modeling/save_gdrn.sh  \
    configs/gdrn/lmCropBlenderSO/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e_SO/resnest50d_a6_AugCosyAAEGary_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e_ape.py 0 \
    output/gdrn/lm_crop_blender/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e/ape/model_final_wo_optim-ab325295.pth \
    DATASETS.TEST="lm_crop_ape_train,"

# 2 benchvise
./core/gdrn_modeling/save_gdrn.sh  \
    configs/gdrn/lmCropBlenderSO/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e_SO/resnest50d_a6_AugCosyAAEGary_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e_benchvise.py 0 \
    output/gdrn/lm_crop_blender/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e/benchvise/model_final_wo_optim-cba7da41.pth \
    DATASETS.TEST="lm_crop_benchvise_train,"

# 4 camera
./core/gdrn_modeling/save_gdrn.sh \
    configs/gdrn/lmCropBlenderSO/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e_SO/resnest50d_a6_AugCosyAAEGary_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e_camera.py 0 \
    output/gdrn/lm_crop_blender/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e/camera/model_final_wo_optim-5055333e.pth \
    DATASETS.TEST="lm_crop_camera_train,"

# 5 can
./core/gdrn_modeling/save_gdrn.sh  \
    configs/gdrn/lmCropBlenderSO/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e_SO/resnest50d_a6_AugCosyAAEGary_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e_can.py    0 \
    output/gdrn/lm_crop_blender/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e/can/model_final_wo_optim-889ae475.pth \
    DATASETS.TEST="lm_crop_can_train,"

# 6 cat
./core/gdrn_modeling/save_gdrn.sh \
    configs/gdrn/lmCropBlenderSO/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e_SO/resnest50d_a6_AugCosyAAEGary_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e_cat.py 0 \
    output/gdrn/lm_crop_blender/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e/cat/model_final_wo_optim-c754e587.pth \
    DATASETS.TEST="lm_crop_cat_train,"

# 8 driller
./core/gdrn_modeling/save_gdrn.sh  \
    configs/gdrn/lmCropBlenderSO/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e_SO/resnest50d_a6_AugCosyAAEGary_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e_driller.py 0 \
    output/gdrn/lm_crop_blender/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e/driller/model_final_wo_optim-c1cc7169.pth \
    DATASETS.TEST="lm_crop_driller_train,"

# 9 duck
./core/gdrn_modeling/save_gdrn.sh  \
    configs/gdrn/lmCropBlenderSO/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e_SO/resnest50d_a6_AugCosyAAEGary_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e_duck.py 0 \
    output/gdrn/lm_crop_blender/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e/duck/model_final_wo_optim-084c3364.pth \
    DATASETS.TEST="lm_crop_duck_train,"

# 12 holepuncher
./core/gdrn_modeling/save_gdrn.sh  \
    configs/gdrn/lmCropBlenderSO/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e_SO/resnest50d_a6_AugCosyAAEGary_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e_holepuncher.py 0 \
    output/gdrn/lm_crop_blender/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e/holepuncher/model_final_wo_optim-86d52ab7.pth \
    DATASETS.TEST="lm_crop_holepuncher_train,"

# 13 iron
./core/gdrn_modeling/save_gdrn.sh  \
    configs/gdrn/lmCropBlenderSO/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e_SO/resnest50d_a6_AugCosyAAEGary_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e_iron.py 0 \
    output/gdrn/lm_crop_blender/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e/iron/model_final_wo_optim-6bdb61b0.pth \
    DATASETS.TEST="lm_crop_iron_train,"

# 14 lamp
./core/gdrn_modeling/save_gdrn.sh  \
    configs/gdrn/lmCropBlenderSO/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e_SO/resnest50d_a6_AugCosyAAEGary_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e_lamp.py 0 \
    output/gdrn/lm_crop_blender/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e/lamp/model_final_wo_optim-bf52eba3.pth \
    DATASETS.TEST="lm_crop_lamp_train,"

# 15 phone
./core/gdrn_modeling/save_gdrn.sh  \
    configs/gdrn/lmCropBlenderSO/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e_SO/resnest50d_a6_AugCosyAAEGary_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e_phone.py 0 \
    output/gdrn/lm_crop_blender/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e/phone/model_final_wo_optim-4172540b.pth \
    DATASETS.TEST="lm_crop_phone_train,"
