#!/usr/bin/env bash
set -ex
# 1 ape
./core/gdrn_modeling/save_gdrn.sh configs/gdrn/lmPbrSingleObj/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e_SO/resnest50d_a6_AugCosyAAEGary_BG05_mlBCE_lm_pbr_100e_ape.py \
    0 \
    output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/ape/model_final_wo_optim-e8c99c96.pth \
    DATASETS.DET_FILES_TEST="datasets/BOP_DATASETS/lm/test/test_bboxes/yolov4x_640_test672_augCosyAAEGray_ranger_lm_pbr_lm_13_all_16e.json," \
    DATASETS.TEST='lm_real_ape_all,'

# 2 benchvise
./core/gdrn_modeling/save_gdrn.sh configs/gdrn/lmPbrSingleObj/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e_SO/resnest50d_a6_AugCosyAAEGary_BG05_mlBCE_lm_pbr_100e_benchvise.py \
    0 \
    output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/benchvise/model_final_wo_optim-85b3563e.pth \
    DATASETS.DET_FILES_TEST="datasets/BOP_DATASETS/lm/test/test_bboxes/yolov4x_640_test672_augCosyAAEGray_ranger_lm_pbr_lm_13_all_16e.json," \
    DATASETS.TEST='lm_real_benchvise_all,'

# 4 camera
./core/gdrn_modeling/save_gdrn.sh configs/gdrn/lmPbrSingleObj/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e_SO/resnest50d_a6_AugCosyAAEGary_BG05_mlBCE_lm_pbr_100e_camera.py \
    0 \
    output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/camera/model_final_wo_optim-1b281dbe.pth \
    DATASETS.DET_FILES_TEST="datasets/BOP_DATASETS/lm/test/test_bboxes/yolov4x_640_test672_augCosyAAEGray_ranger_lm_pbr_lm_13_all_16e.json," \
    DATASETS.TEST='lm_real_camera_all,'

# 5 can
./core/gdrn_modeling/save_gdrn.sh configs/gdrn/lmPbrSingleObj/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e_SO/resnest50d_a6_AugCosyAAEGary_BG05_mlBCE_lm_pbr_100e_can.py \
    0 \
    output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/can/model_final_wo_optim-53ea56ee.pth \
    DATASETS.DET_FILES_TEST="datasets/BOP_DATASETS/lm/test/test_bboxes/yolov4x_640_test672_augCosyAAEGray_ranger_lm_pbr_lm_13_all_16e.json," \
    DATASETS.TEST='lm_real_can_all,'

# 6 cat
./core/gdrn_modeling/save_gdrn.sh configs/gdrn/lmPbrSingleObj/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e_SO/resnest50d_a6_AugCosyAAEGary_BG05_mlBCE_lm_pbr_100e_cat.py \
    0 \
    output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/cat/model_final_wo_optim-f38cfafd.pth \
    DATASETS.DET_FILES_TEST="datasets/BOP_DATASETS/lm/test/test_bboxes/yolov4x_640_test672_augCosyAAEGray_ranger_lm_pbr_lm_13_all_16e.json," \
    DATASETS.TEST='lm_real_cat_all,'

# 8 driller
./core/gdrn_modeling/save_gdrn.sh configs/gdrn/lmPbrSingleObj/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e_SO/resnest50d_a6_AugCosyAAEGary_BG05_mlBCE_lm_pbr_100e_driller.py \
    0 \
    output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/driller/model_final_wo_optim-4cfc7d64.pth \
    DATASETS.DET_FILES_TEST="datasets/BOP_DATASETS/lm/test/test_bboxes/yolov4x_640_test672_augCosyAAEGray_ranger_lm_pbr_lm_13_all_16e.json," \
    DATASETS.TEST='lm_real_driller_all,'

# 9 duck
./core/gdrn_modeling/save_gdrn.sh configs/gdrn/lmPbrSingleObj/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e_SO/resnest50d_a6_AugCosyAAEGary_BG05_mlBCE_lm_pbr_100e_duck.py \
    0 \
    output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/duck/model_final_wo_optim-0bde58bb.pth \
    DATASETS.DET_FILES_TEST="datasets/BOP_DATASETS/lm/test/test_bboxes/yolov4x_640_test672_augCosyAAEGray_ranger_lm_pbr_lm_13_all_16e.json," \
    DATASETS.TEST='lm_real_duck_all,'

# 10 eggbox
./core/gdrn_modeling/save_gdrn.sh configs/gdrn/lmPbrSingleObj/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e_SO/resnest50d_a6_AugCosyAAEGary_BG05_mlBCE_lm_pbr_100e_eggbox_Rsym.py \
    0 \
    output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/eggbox_Rsym/model_final_wo_optim-d0656ca7.pth \
    DATASETS.DET_FILES_TEST="datasets/BOP_DATASETS/lm/test/test_bboxes/yolov4x_640_test672_augCosyAAEGray_ranger_lm_pbr_lm_13_all_16e.json," \
    DATASETS.TEST='lm_real_eggbox_all,'

# 11 glue
./core/gdrn_modeling/save_gdrn.sh configs/gdrn/lmPbrSingleObj/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e_SO/resnest50d_a6_AugCosyAAEGary_BG05_mlBCE_lm_pbr_100e_glue_Rsym.py \
    0 \
    output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/glue_Rsym/model_final_wo_optim-324d8f16.pth \
    DATASETS.DET_FILES_TEST="datasets/BOP_DATASETS/lm/test/test_bboxes/yolov4x_640_test672_augCosyAAEGray_ranger_lm_pbr_lm_13_all_16e.json," \
    DATASETS.TEST='lm_real_glue_all,'

# 12 holepuncher
./core/gdrn_modeling/save_gdrn.sh configs/gdrn/lmPbrSingleObj/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e_SO/resnest50d_a6_AugCosyAAEGary_BG05_mlBCE_lm_pbr_100e_holepuncher.py \
    0 \
    output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/holepuncher/model_final_wo_optim-eab19662.pth \
    DATASETS.DET_FILES_TEST="datasets/BOP_DATASETS/lm/test/test_bboxes/yolov4x_640_test672_augCosyAAEGray_ranger_lm_pbr_lm_13_all_16e.json," \
    DATASETS.TEST='lm_real_holepuncher_all,'

# 13 iron
./core/gdrn_modeling/save_gdrn.sh configs/gdrn/lmPbrSingleObj/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e_SO/resnest50d_a6_AugCosyAAEGary_BG05_mlBCE_lm_pbr_100e_iron.py \
    0 \
    output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/iron/model_final_wo_optim-025a740e.pth \
    DATASETS.DET_FILES_TEST="datasets/BOP_DATASETS/lm/test/test_bboxes/yolov4x_640_test672_augCosyAAEGray_ranger_lm_pbr_lm_13_all_16e.json," \
    DATASETS.TEST='lm_real_iron_all,'

# 14 lamp
./core/gdrn_modeling/save_gdrn.sh  configs/gdrn/lmPbrSingleObj/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e_SO/resnest50d_a6_AugCosyAAEGary_BG05_mlBCE_lm_pbr_100e_lamp.py \
    0 \
    output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/lamp/model_final_wo_optim-34042758.pth \
    DATASETS.DET_FILES_TEST="datasets/BOP_DATASETS/lm/test/test_bboxes/yolov4x_640_test672_augCosyAAEGray_ranger_lm_pbr_lm_13_all_16e.json," \
    DATASETS.TEST='lm_real_lamp_all,'

# 15 phone
./core/gdrn_modeling/save_gdrn.sh configs/gdrn/lmPbrSingleObj/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e_SO/resnest50d_a6_AugCosyAAEGary_BG05_mlBCE_lm_pbr_100e_phone.py \
    0 \
    output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/phone/model_final_wo_optim-525a29f8.pth \
    DATASETS.DET_FILES_TEST="datasets/BOP_DATASETS/lm/test/test_bboxes/yolov4x_640_test672_augCosyAAEGray_ranger_lm_pbr_lm_13_all_16e.json," \
    DATASETS.TEST='lm_real_phone_all,'
