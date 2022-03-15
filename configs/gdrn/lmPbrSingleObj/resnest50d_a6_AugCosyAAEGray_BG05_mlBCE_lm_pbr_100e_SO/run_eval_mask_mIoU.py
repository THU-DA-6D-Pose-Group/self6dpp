import os

id2obj = {
    1: "ape",
    2: "benchvise",
    # 3: "bowl",
    4: "camera",
    5: "can",
    6: "cat",
    # 7: "cup",
    8: "driller",
    9: "duck",
    10: "eggbox",
    11: "glue",
    12: "holepuncher",
    13: "iron",
    14: "lamp",
    15: "phone",
}

res_paths = [
    "output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/ape/inference_model_final_wo_optim-e8c99c96/lm_real_ape_test/results.pkl",
    "output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/benchvise/inference_model_final_wo_optim-85b3563e/lm_real_benchvise_test/results.pkl",
    "output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/camera/inference_model_final_wo_optim-1b281dbe/lm_real_camera_test/results.pkl",
    "output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/can/inference_model_final_wo_optim-53ea56ee/lm_real_can_test/results.pkl",
    "output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/cat/inference_model_final_wo_optim-f38cfafd/lm_real_cat_test/results.pkl",
    "output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/driller/inference_model_final_wo_optim-4cfc7d64/lm_real_driller_test/results.pkl",
    "output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/duck/inference_model_final_wo_optim-0bde58bb/lm_real_duck_test/results.pkl",
    "output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/eggbox_Rsym/inference_model_final_wo_optim-d0656ca7/lm_real_eggbox_test/results.pkl",
    "output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/glue_Rsym/inference_model_final_wo_optim-324d8f16/lm_real_glue_test/results.pkl",
    "output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/holepuncher/inference_model_final_wo_optim-eab19662/lm_real_holepuncher_test/results.pkl",
    "output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/iron/inference_model_final_wo_optim-025a740e/lm_real_iron_test/results.pkl",
    "output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/lamp/inference_model_final_wo_optim-34042758/lm_real_lamp_test/results.pkl",
    "output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/phone/inference_model_final_wo_optim-525a29f8/lm_real_phone_test/results.pkl",
]

obj_names = list(id2obj.values())
for obj, res_path in zip(obj_names, res_paths):
    cmd = (
        "python core/self6dpp/tools/compute_mIoU_mask.py " "--dataset lm_real_{}_test --cls {} " "--res_path {}"
    ).format(obj, obj, res_path)
    print(cmd)
    os.system(cmd)
