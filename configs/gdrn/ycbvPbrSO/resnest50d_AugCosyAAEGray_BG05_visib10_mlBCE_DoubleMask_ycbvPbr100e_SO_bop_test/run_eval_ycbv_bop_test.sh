#!/usr/bin/env bash
set -ex

# 1
./core/gdrn_modeling/test_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_bop_test/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_bop_test_01_02MasterChefCan.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/01_02MasterChefCan/model_final_wo_optim-8624c7f1.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 2
./core/gdrn_modeling/test_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_bop_test/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_bop_test_02_03CrackerBox.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/02_03CrackerBox/model_final_wo_optim-72f9c1da.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 3
./core/gdrn_modeling/test_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_bop_test/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_bop_test_03_04SugarBox.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/03_04SugarBox/model_final_wo_optim-bf2dc932.pth \
  VAL.SAVE_BOP_CSV_ONLY=True


# 4
./core/gdrn_modeling/test_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_bop_test/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_bop_test_04_05TomatoSoupCan.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/04_05TomatoSoupCan/model_final_wo_optim-b142bb56.pth \
  VAL.SAVE_BOP_CSV_ONLY=True


# 5
./core/gdrn_modeling/test_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_bop_test/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_bop_test_05_06MustardBottle.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/05_06MustardBottle/model_final_wo_optim-86dde7e2.pth \
  VAL.SAVE_BOP_CSV_ONLY=True


# 6
./core/gdrn_modeling/test_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_bop_test/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_bop_test_06_07TunaFishCan.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/06_07TunaFishCan/model_final_wo_optim-0b376921.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 7
./core/gdrn_modeling/test_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_bop_test/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_bop_test_07_08PuddingBox.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/07_08PuddingBox/model_final_wo_optim-23fb01c0.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 8
./core/gdrn_modeling/test_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_bop_test/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_bop_test_08_09GelatinBox.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/08_09GelatinBox/model_final_wo_optim-1256bdbd.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 9
./core/gdrn_modeling/test_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_bop_test/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_bop_test_09_10PottedMeatCan.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/09_10PottedMeatCan/model_final_wo_optim-cc1232e2.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 10
./core/gdrn_modeling/test_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_bop_test/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_bop_test_10_11Banana.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/10_11Banana/model_final_wo_optim-427d6321.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 11
./core/gdrn_modeling/test_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_bop_test/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_bop_test_11_19PitcherBase.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/11_19PitcherBase/model_final_wo_optim-f052dafc.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 12
./core/gdrn_modeling/test_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_bop_test/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_bop_test_12_21BleachCleanser.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/12_21BleachCleanser/model_final_wo_optim-59a61f06.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 13
./core/gdrn_modeling/test_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_bop_test/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_bop_test_13_24Bowl.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/13_24Bowl/model_final_wo_optim-55b952fc.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 14
./core/gdrn_modeling/test_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_bop_test/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_bop_test_14_25Mug.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/14_25Mug/model_final_wo_optim-6b280ec5.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 15
./core/gdrn_modeling/test_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_bop_test/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_bop_test_15_35PowerDrill.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/15_35PowerDrill/model_final_wo_optim-0769bee7.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 16
./core/gdrn_modeling/test_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_bop_test/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_bop_test_16_36WoodBlock.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/16_36WoodBlock/model_final_wo_optim-baaa72c3.pth \
  VAL.SAVE_BOP_CSV_ONLY=True


# 17
./core/gdrn_modeling/test_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_bop_test/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_bop_test_17_37Scissors.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/17_37Scissors/model_final_wo_optim-eb042de2.pth \
  VAL.SAVE_BOP_CSV_ONLY=True


# 18
./core/gdrn_modeling/test_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_bop_test/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_bop_test_18_40LargeMarker.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/18_40LargeMarker/model_final_wo_optim-3c805088.pth \
  VAL.SAVE_BOP_CSV_ONLY=True


# 19
./core/gdrn_modeling/test_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_bop_test/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_bop_test_19_51LargeClamp.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/19_51LargeClamp/model_final_wo_optim-9643daa1.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 20
./core/gdrn_modeling/test_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_bop_test/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_bop_test_20_52ExtraLargeClamp.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/20_52ExtraLargeClamp/model_final_wo_optim-82f2dafa.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 21
./core/gdrn_modeling/test_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_bop_test/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_bop_test_21_61FoamBrick.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/21_61FoamBrick/model_final_wo_optim-375a4cba.pth \
  VAL.SAVE_BOP_CSV_ONLY=True


# eval merged csv
#python core/gdrn_modeling/engine/test_utils.py \
#    --result_dir output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/merged-ycbv-bop-test/ \
#    --result_names resnest50d-AugCosyAAEGray-BG05-visib10-mlBCE-DoubleMask-ycbvPbr100e-SO-bop-test-merged-test-iter0_ycbv-test.csv \
#    --dataset ycbv \
#    --split test \
#    --split-type "" \
#    --targets_name test_targets_bop19.json \
#    --error_types mspd,mssd,vsd,reS,teS,reteS,ad \
#    --render_type cpp

#objects           002_master_chef_can  003_cracker_box  004_sugar_box  005_tomato_soup_can  006_mustard_bottle  007_tuna_fish_can  008_pudding_box  009_gelatin_box  010_potted_meat_can  011_banana  019_pitcher_base  021_bleach_cleanser  024_bowl  025_mug  035_power_drill  036_wood_block  037_scissors  040_large_marker  051_large_clamp  052_extra_large_clamp  061_foam_brick  Avg(21)
#mspd_5:50         80.67                86.93            87.44          84.55                89.67               91.30              92.00            93.73            67.24                81.80       85.47             73.83                77.73     88.93    84.93            61.87           84.00         92.13             55.47            43.47                  92.53           81.25
#mssd_0.050:0.500  62.97                86.76            79.49          55.13                89.80               56.03              82.93            53.47            54.22                85.67       87.60             79.43                55.13     46.47    82.63            47.60           50.67         55.33             46.60            64.07                  57.07           67.57
#vsd_0.050:0.500   60.43                81.33            72.68          47.38                80.84               50.84              74.74            50.44            54.86                65.07       83.07             72.22                44.17     38.81    72.77            39.64           33.33         37.86             8.91             24.11                  49.19           57.78
#reS_2             7.67                 38.22            18.13          11.38                41.33               19.67              22.67            17.33            1.33                 6.67        10.67             4.67                 16.00     6.00     10.33            4.00            42.67         36.67             10.00            5.33                   21.33           16.77
#reS_5             48.67                88.00            94.93          58.26                98.00               50.67              98.67            90.67            39.56                42.67       80.44             37.00                59.33     65.33    99.00            33.33           76.00         72.67             59.33            46.67                  81.33           67.64
#reS_10            84.67                100.00           100.00         94.64                100.00              100.00             100.00           100.00           74.67                80.67       99.56             79.33                90.00     96.00    100.00           93.33           100.00        100.00            80.67            83.33                  97.33           93.06
#teS_2             28.33                49.78            46.40          17.41                84.67               48.00              85.33            21.33            55.11                89.33       75.11             60.67                20.00     3.33     54.67            17.33           4.00          40.67             17.33            72.00                  50.67           44.83
#teS_5             88.33                100.00           100.00         95.31                100.00              100.00             100.00           100.00           94.22                100.00      99.56             93.67                84.00     98.00    100.00           45.33           46.67         90.00             80.00            84.00                  96.00           90.24
#teS_10            100.00               100.00           100.00         96.43                100.00              100.00             100.00           100.00           96.00                100.00      99.56             98.67                97.33     100.00   100.00           84.00           98.67         100.00            80.67            84.00                  100.00          96.92
#reteS_2           6.33                 14.67            10.67          4.02                 31.33               8.67               22.67            5.33             0.44                 5.33        5.78              3.00                 4.00      0.00     3.00             1.33            1.33          7.33              0.00             0.67                   14.67           7.17
#reteS_5           40.33                88.00            94.93          58.26                98.00               50.67              98.67            90.67            39.56                42.67       80.44             36.67                52.00     63.33    99.00            17.33           46.67         64.00             58.67            46.67                  78.67           64.06
#reteS_10          84.67                100.00           100.00         94.64                100.00              100.00             100.00           100.00           74.67                80.67       99.56             79.00                90.00     96.00    100.00           78.67           98.67         100.00            80.67            83.33                  97.33           92.28
#ad_0.020          2.33                 0.89             1.07           0.00                 2.00                0.33               4.00             0.00             0.00                 0.67        1.33              2.67                 4.67      0.00     0.67             1.33            0.00          0.67              4.67             7.33                   0.00            1.65
#ad_0.050          27.00                16.00            8.27           0.67                 40.67               8.67               22.67            0.00             14.22                28.67       16.44             36.67                10.00     0.00     19.00            20.00           0.00          13.33             20.67            60.67                  21.33           18.33
#ad_0.100          63.33                82.22            44.00          6.92                 82.00               24.00              52.00            0.00             30.67                90.67       95.11             68.33                33.33     21.33    63.67            44.00           4.00          50.00             34.67            84.00                  52.00           48.87
