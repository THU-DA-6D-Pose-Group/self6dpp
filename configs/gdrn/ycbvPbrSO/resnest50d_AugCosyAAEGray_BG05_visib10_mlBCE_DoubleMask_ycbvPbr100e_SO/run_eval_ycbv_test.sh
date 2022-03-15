#!/usr/bin/env bash
set -ex

# 1
./core/gdrn_modeling/test_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_01_02MasterChefCan.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/01_02MasterChefCan/model_final_wo_optim-8624c7f1.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 2
./core/gdrn_modeling/test_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_02_03CrackerBox.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/02_03CrackerBox/model_final_wo_optim-72f9c1da.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 3
./core/gdrn_modeling/test_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_03_04SugarBox.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/03_04SugarBox/model_final_wo_optim-bf2dc932.pth \
  VAL.SAVE_BOP_CSV_ONLY=True


# 4
./core/gdrn_modeling/test_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_04_05TomatoSoupCan.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/04_05TomatoSoupCan/model_final_wo_optim-b142bb56.pth \
  VAL.SAVE_BOP_CSV_ONLY=True


# 5
./core/gdrn_modeling/test_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_05_06MustardBottle.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/05_06MustardBottle/model_final_wo_optim-86dde7e2.pth \
  VAL.SAVE_BOP_CSV_ONLY=True


# 6
./core/gdrn_modeling/test_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_06_07TunaFishCan.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/06_07TunaFishCan/model_final_wo_optim-0b376921.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 7
./core/gdrn_modeling/test_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_07_08PuddingBox.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/07_08PuddingBox/model_final_wo_optim-23fb01c0.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 8
./core/gdrn_modeling/test_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_08_09GelatinBox.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/08_09GelatinBox/model_final_wo_optim-1256bdbd.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 9
./core/gdrn_modeling/test_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_09_10PottedMeatCan.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/09_10PottedMeatCan/model_final_wo_optim-cc1232e2.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 10
./core/gdrn_modeling/test_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_10_11Banana.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/10_11Banana/model_final_wo_optim-427d6321.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 11
./core/gdrn_modeling/test_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_11_19PitcherBase.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/11_19PitcherBase/model_final_wo_optim-f052dafc.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 12
./core/gdrn_modeling/test_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_12_21BleachCleanser.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/12_21BleachCleanser/model_final_wo_optim-59a61f06.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 13
./core/gdrn_modeling/test_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_13_24Bowl.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/13_24Bowl/model_final_wo_optim-55b952fc.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 14
./core/gdrn_modeling/test_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_14_25Mug.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/14_25Mug/model_final_wo_optim-6b280ec5.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 15
./core/gdrn_modeling/test_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_15_35PowerDrill.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/15_35PowerDrill/model_final_wo_optim-0769bee7.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 16
./core/gdrn_modeling/test_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_16_36WoodBlock.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/16_36WoodBlock/model_final_wo_optim-baaa72c3.pth \
  VAL.SAVE_BOP_CSV_ONLY=True


# 17
./core/gdrn_modeling/test_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_17_37Scissors.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/17_37Scissors/model_final_wo_optim-eb042de2.pth \
  VAL.SAVE_BOP_CSV_ONLY=True


# 18
./core/gdrn_modeling/test_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_18_40LargeMarker.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/18_40LargeMarker/model_final_wo_optim-3c805088.pth \
  VAL.SAVE_BOP_CSV_ONLY=True


# 19
./core/gdrn_modeling/test_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_19_51LargeClamp.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/19_51LargeClamp/model_final_wo_optim-9643daa1.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 20
./core/gdrn_modeling/test_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_20_52ExtraLargeClamp.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/20_52ExtraLargeClamp/model_final_wo_optim-82f2dafa.pth \
  VAL.SAVE_BOP_CSV_ONLY=True

# 21
./core/gdrn_modeling/test_gdrn.sh \
  configs/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO_21_61FoamBrick.py 1 \
  output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/21_61FoamBrick/model_final_wo_optim-375a4cba.pth \
  VAL.SAVE_BOP_CSV_ONLY=True


# eval merged csv
#python core/gdrn_modeling/engine/test_utils.py \
#    --result_dir output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/merged-ycbv-test/ \
#    --result_names resnest50d-AugCosyAAEGray-BG05-visib10-mlBCE-DoubleMask-ycbvPbr100e-SO-merged-test-iter0_ycbvposecnn-test.csv \
#    --dataset ycbvposecnn \
#    --split test \
#    --split-type "" \
#    --targets_name ycbv_test_targets_keyframe.json \
#    --error_types ad,AUCad,AUCadi,reteS,reS,teS,projS \
#    --render_type cpp
#objects      002_master_chef_can  003_cracker_box  004_sugar_box  005_tomato_soup_can  006_mustard_bottle  007_tuna_fish_can  008_pudding_box  009_gelatin_box  010_potted_meat_can  011_banana  019_pitcher_base  021_bleach_cleanser  024_bowl  025_mug  035_power_drill  036_wood_block  037_scissors  040_large_marker  051_large_clamp  052_extra_large_clamp  061_foam_brick  Avg(21)
#ad_0.020     0.00                 4.72             0.76           0.00                 1.96                0.26               5.61             0.00             0.13                 2.11        3.86              2.33                 1.72      0.00     1.80             2.48            0.00          0.00              2.39             8.65                   0.35            1.86
#ad_0.050     0.00                 22.81            6.09           0.56                 37.82               6.10               22.43            0.47             9.01                 37.20       24.39             37.41                5.42      0.00     20.62            23.14           0.00          0.00              25.00            62.76                  23.96           17.39
#ad_0.100     0.00                 83.99            41.29          5.97                 82.07               21.78              44.39            4.67             21.02                90.77       97.72             69.97                20.94     0.00     58.75            47.93           2.21          0.15              40.59            87.39                  61.11           42.03
#AUCad_1:10   9.62                 85.63            82.40          77.45                91.99               86.74              89.67            78.93            74.67                92.88       89.16             80.31                77.59     73.13    84.55            76.78           55.64         70.45             76.52            84.57                  94.44           77.77
#AUCadi_1:10  89.49                94.50            93.15          89.83                97.51               95.98              96.07            90.19            90.44                99.16       97.84             90.54                77.59     90.11    94.73            76.78           74.09         82.89             76.52            84.57                  94.44           89.35
#reteS_2      4.77                 11.75            10.32          2.85                 31.09               7.84               19.16            15.42            0.91                 3.69        7.02              4.96                 2.71      0.16     4.82             0.83            1.10          11.73             0.00             1.91                   6.25            7.11
#reteS_5      36.58                86.29            96.95          66.81                87.68               57.49              89.25            92.99            44.39                49.60       87.02             38.68                49.01     62.58    96.12            21.90           52.49         61.88             55.06            50.29                  80.90           64.95
#reteS_10     78.83                100.00           100.00         96.94                99.44               100.00             100.00           100.00           81.98                89.45       99.47             80.66                80.30     90.72    100.00           78.51           98.90         94.14             85.96            82.40                  99.31           92.24
#reS_2        7.95                 28.23            19.12          9.31                 40.34               15.77              19.63            37.38            1.96                 6.33        10.00             5.93                 16.01     3.93     13.34            6.20            40.88         41.05             9.69             5.13                   15.97           16.86
#reS_5        44.63                86.29            97.04          66.94                87.68               57.49              89.25            92.99            44.39                49.60       87.02             40.23                58.87     63.21    96.22            42.15           71.27         68.83             56.18            50.29                  82.64           68.25
#reS_10       78.83                100.00           100.00         96.94                99.44               100.00             100.00           100.00           81.98                89.45       99.47             83.09                80.30     90.72    100.00           88.84           100.00        94.14             85.96            82.40                  99.31           92.90
#teS_2        25.05                48.85            42.72          17.71                85.15               56.62              74.30            24.30            44.91                89.18       82.46             62.29                10.59     4.87     52.13            18.60           2.21          33.80             25.28            68.18                  54.17           43.97
#teS_5        88.97                100.00           99.92          97.22                100.00              100.00             100.00           100.00           95.69                100.00      99.47             90.96                75.62     99.37    99.91            51.65           54.70         83.80             87.78            88.12                  97.92           91.00
#teS_10       100.00               100.00           100.00         98.47                100.00              100.00             100.00           100.00           96.48                100.00      99.47             95.63                92.36     100.00   100.00           89.26           98.90         97.38             89.04            88.12                  100.00          97.39
#projS_2      2.49                 1.15             1.10           1.39                 1.40                5.05               6.07             19.63            0.78                 0.00        0.00              0.00                 0.00      0.79     0.09             0.00            0.00          2.78              0.00             0.00                   6.60            2.35
#projS_5      32.31                64.98            69.88          64.72                64.43               84.67              85.51            94.86            62.01                31.40       72.28             10.30                5.17      66.19    58.47            4.13            56.91         78.86             1.83             7.04                   86.11           52.48
#projS_10     59.84                98.50            98.05          95.76                97.48               99.91              100.00           100.00           83.03                92.61       99.47             76.97                78.82     100.00   99.81            38.84           100.00        99.85             22.33            30.94                  100.00          84.39
