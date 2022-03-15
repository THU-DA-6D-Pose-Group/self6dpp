#!/usr/bin/env bash
set -ex

# benchvise
./core/deepim/save_deepim.sh \
  configs/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_02_benchvise.py 1 \
  output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/benchvise/model_final_wo_optim-956eb2eb.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/hb_bench_driller_phone/init_poses/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e_so_GdrnPose_with_yolov4_pbr_bbox_hbBdpAll.json," \
  DATASETS.TEST="hb_bdp_benchvise_all_lmK,"

# driller
./core/deepim/save_deepim.sh \
  configs/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_08_driller.py 1 \
  output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/driller/model_final_wo_optim-45c5dd36.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/hb_bench_driller_phone/init_poses/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e_so_GdrnPose_with_yolov4_pbr_bbox_hbBdpAll.json," \
  DATASETS.TEST="hb_bdp_driller_all_lmK,"


# phone
./core/deepim/save_deepim.sh \
  configs/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_15_phone.py 1 \
  output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/phone/model_final_wo_optim-61aad172.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/hb_bench_driller_phone/init_poses/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e_so_GdrnPose_with_yolov4_pbr_bbox_hbBdpAll.json," \
  DATASETS.TEST="hb_bdp_phone_all_lmK,"

