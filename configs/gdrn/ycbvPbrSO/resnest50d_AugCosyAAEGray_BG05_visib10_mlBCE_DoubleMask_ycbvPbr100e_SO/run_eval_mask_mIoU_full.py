import os

id2obj = {
    1: "002_master_chef_can",  # [1.3360, -0.5000, 3.5105]
    2: "003_cracker_box",  # [0.5575, 1.7005, 4.8050]
    3: "004_sugar_box",  # [-0.9520, 1.4670, 4.3645]
    4: "005_tomato_soup_can",  # [-0.0240, -1.5270, 8.4035]
    5: "006_mustard_bottle",  # [1.2995, 2.4870, -11.8290]
    6: "007_tuna_fish_can",  # [-0.1565, 0.1150, 4.2625]
    7: "008_pudding_box",  # [1.1645, -4.2015, 3.1190]
    8: "009_gelatin_box",  # [1.4460, -0.5915, 3.6085]
    9: "010_potted_meat_can",  # [2.4195, 0.3075, 8.0715]
    10: "011_banana",  # [-18.6730, 12.1915, -1.4635]
    11: "019_pitcher_base",  # [5.3370, 5.8855, 25.6115]
    12: "021_bleach_cleanser",  # [4.9290, -2.4800, -13.2920]
    13: "024_bowl",  # [-0.2270, 0.7950, -2.9675]
    14: "025_mug",  # [-8.4675, -0.6995, -1.6145]
    15: "035_power_drill",  # [9.0710, 20.9360, -2.1190]
    16: "036_wood_block",  # [1.4265, -2.5305, 17.1890]
    17: "037_scissors",  # [7.0535, -28.1320, 0.0420]
    18: "040_large_marker",  # [0.0460, -2.1040, 0.3500]
    19: "051_large_clamp",  # [10.5180, -1.9640, -0.4745]
    20: "052_extra_large_clamp",  # [-0.3950, -10.4130, 0.1620]
    21: "061_foam_brick",  # [-0.0805, 0.0805, -8.2435]
}

res_paths = [
    "output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/01_02MasterChefCan/inference_model_final_wo_optim-8624c7f1/ycbv_test/results.pkl",
    "output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/02_03CrackerBox/inference_model_final_wo_optim-72f9c1da/ycbv_test/results.pkl",
    "output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/03_04SugarBox/inference_model_final_wo_optim-bf2dc932/ycbv_test/results.pkl",
    "output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/04_05TomatoSoupCan/inference_model_final_wo_optim-b142bb56/ycbv_test/results.pkl",
    "output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/05_06MustardBottle/inference_model_final_wo_optim-86dde7e2/ycbv_test/results.pkl",
    "output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/06_07TunaFishCan/inference_model_final_wo_optim-0b376921/ycbv_test/results.pkl",
    "output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/07_08PuddingBox/inference_model_final_wo_optim-23fb01c0/ycbv_test/results.pkl",
    "output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/08_09GelatinBox/inference_model_final_wo_optim-1256bdbd/ycbv_test/results.pkl",
    "output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/09_10PottedMeatCan/inference_model_final_wo_optim-cc1232e2/ycbv_test/results.pkl",
    "output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/10_11Banana/inference_model_final_wo_optim-427d6321/ycbv_test/results.pkl",
    "output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/11_19PitcherBase/inference_model_final_wo_optim-f052dafc/ycbv_test/results.pkl",
    "output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/12_21BleachCleanser/inference_model_final_wo_optim-59a61f06/ycbv_test/results.pkl",
    "output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/13_24Bowl/inference_model_final_wo_optim-55b952fc/ycbv_test/results.pkl",
    "output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/14_25Mug/inference_model_final_wo_optim-6b280ec5/ycbv_test/results.pkl",
    "output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/15_35PowerDrill/inference_model_final_wo_optim-0769bee7/ycbv_test/results.pkl",
    "output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/16_36WoodBlock/inference_model_final_wo_optim-baaa72c3/ycbv_test/results.pkl",
    "output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/17_37Scissors/inference_model_final_wo_optim-eb042de2/ycbv_test/results.pkl",
    "output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/18_40LargeMarker/inference_model_final_wo_optim-3c805088/ycbv_test/results.pkl",
    "output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/19_51LargeClamp/inference_model_final_wo_optim-9643daa1/ycbv_test/results.pkl",
    "output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/20_52ExtraLargeClamp/inference_model_final_wo_optim-82f2dafa/ycbv_test/results.pkl",
    "output/gdrn/ycbvPbrSO/resnest50d_AugCosyAAEGray_BG05_visib10_mlBCE_DoubleMask_ycbvPbr100e_SO/21_61FoamBrick/inference_model_final_wo_optim-375a4cba/ycbv_test/results.pkl",
]


obj_names = list(id2obj.values())
for obj, res_path in zip(obj_names, res_paths):
    cmd = (
        "python core/self6dpp/tools/compute_mIoU_mask.py "
        "--dataset ycbv_{}_test --cls {} "
        "--res_path {}  --mask_type full_mask"
    ).format(obj, obj, res_path)
    print(cmd)
    os.system(cmd)
