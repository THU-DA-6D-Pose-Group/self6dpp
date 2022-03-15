#!/usr/bin/env bash
set -ex

# benchvise
./core/deepim/test_deepim.sh \
  configs/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_02_benchvise.py 1 \
  output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/benchvise/model_final_wo_optim-956eb2eb.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/hb_bench_driller_phone/init_poses/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e_so_GdrnPose_with_yolov4_pbr_bbox_hbBdpAll.json," \
  DATASETS.TEST="hb_bdp_benchvise_test_lmK,"

# driller
./core/deepim/test_deepim.sh \
  configs/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_08_driller.py 1 \
  output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/driller/model_final_wo_optim-45c5dd36.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/hb_bench_driller_phone/init_poses/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e_so_GdrnPose_with_yolov4_pbr_bbox_hbBdpAll.json," \
  DATASETS.TEST="hb_bdp_driller_test_lmK,"


# phone
./core/deepim/test_deepim.sh \
  configs/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_Pbr_15_phone.py 1 \
  output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_AggressiveV2_Flat_lmPbr_SO/phone/model_final_wo_optim-61aad172.pth \
  DATASETS.INIT_POSE_FILES_TEST="datasets/hb_bench_driller_phone/init_poses/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e_so_GdrnPose_with_yolov4_pbr_bbox_hbBdpAll.json," \
  DATASETS.TEST="hb_bdp_phone_test_lmK,"



#objects  benchvise  Avg(1)
#ad_2     2.94       2.94
#ad_5     15.65      15.65
#ad_10    52.59      52.59
#rete_2   7.29       7.29
#rete_5   82.47      82.47
#rete_10  96.94      96.94
#re_2     10.35      10.35
#re_5     82.59      82.59
#re_10    96.94      96.94
#te_2     39.88      39.88
#te_5     97.18      97.18
#te_10    98.00      98.00
#proj_2   13.29      13.29
#proj_5   96.24      96.24
#proj_10  97.29      97.29
#re       5.31       5.31
#te       0.03       0.03

#objects  driller  Avg(1)
#ad_2     45.65    45.65
#ad_5     93.65    93.65
#ad_10    98.94    98.94
#rete_2   90.94    90.94
#rete_5   98.12    98.12
#rete_10  98.94    98.94
#re_2     91.06    91.06
#re_5     98.12    98.12
#re_10    98.94    98.94
#te_2     98.47    98.47
#te_5     99.06    99.06
#te_10    99.06    99.06
#proj_2   86.24    86.24
#proj_5   98.94    98.94
#proj_10  99.06    99.06
#re       2.37     2.37
#te       0.01     0.01

#objects  phone  Avg(1)
#ad_2     15.65  15.65
#ad_5     61.29  61.29
#ad_10    88.00  88.00
#rete_2   27.29  27.29
#rete_5   89.88  89.88
#rete_10  97.53  97.53
#re_2     30.82  30.82
#re_5     90.00  90.00
#re_10    98.00  98.00
#te_2     86.59  86.59
#te_5     97.29  97.29
#te_10    98.00  98.00
#proj_2   55.29  55.29
#proj_5   97.41  97.41
#proj_10  98.24  98.24
#re       4.44   4.44
#te       0.01   0.01