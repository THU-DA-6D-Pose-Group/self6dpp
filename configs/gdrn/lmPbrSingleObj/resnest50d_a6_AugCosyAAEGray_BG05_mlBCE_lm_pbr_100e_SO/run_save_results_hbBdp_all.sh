#!/usr/bin/env bash
set -ex

# benchvise
./core/gdrn_modeling/save_gdrn.sh \
  configs/gdrn/lmPbrSingleObj/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e_SO/resnest50d_a6_AugCosyAAEGary_BG05_mlBCE_lm_pbr_100e_benchvise.py 1 \
  output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/benchvise/model_final_wo_optim-85b3563e.pth \
  DATASETS.DET_FILES_TEST="datasets/hb_bench_driller_phone/test_bboxes/yolov4x_640_augCosyAAEGray_ranger_lm_pbr_hb_bdp_all.json," \
  DATASETS.TEST="hb_bdp_benchvise_all,"

# driller
./core/gdrn_modeling/save_gdrn.sh \
  configs/gdrn/lmPbrSingleObj/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e_SO/resnest50d_a6_AugCosyAAEGary_BG05_mlBCE_lm_pbr_100e_driller.py 1 \
  output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/driller/model_final_wo_optim-4cfc7d64.pth \
  DATASETS.DET_FILES_TEST="datasets/hb_bench_driller_phone/test_bboxes/yolov4x_640_augCosyAAEGray_ranger_lm_pbr_hb_bdp_all.json," \
  DATASETS.TEST="hb_bdp_driller_all,"

# phone
./core/gdrn_modeling/save_gdrn.sh \
  configs/gdrn/lmPbrSingleObj/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e_SO/resnest50d_a6_AugCosyAAEGary_BG05_mlBCE_lm_pbr_100e_phone.py 1 \
  output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/phone/model_final_wo_optim-525a29f8.pth \
  DATASETS.DET_FILES_TEST="datasets/hb_bench_driller_phone/test_bboxes/yolov4x_640_augCosyAAEGray_ranger_lm_pbr_hb_bdp_all.json," \
  DATASETS.TEST="hb_bdp_phone_all,"
