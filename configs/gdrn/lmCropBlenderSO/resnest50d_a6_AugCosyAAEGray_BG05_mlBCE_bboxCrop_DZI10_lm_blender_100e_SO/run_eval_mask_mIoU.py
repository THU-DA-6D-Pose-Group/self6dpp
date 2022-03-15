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
    # 10: "eggbox",
    # 11: "glue",
    12: "holepuncher",
    13: "iron",
    14: "lamp",
    15: "phone",
}

res_paths = [
    "output/gdrn/lm_crop_blender/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e/ape/inference_model_final_wo_optim-ab325295/lm_crop_ape_test/results.pkl",
    "output/gdrn/lm_crop_blender/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e/benchvise/inference_model_final_wo_optim-cba7da41/lm_crop_benchvise_test/results.pkl",
    "output/gdrn/lm_crop_blender/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e/camera/inference_model_final_wo_optim-5055333e/lm_crop_camera_test/results.pkl",
    "output/gdrn/lm_crop_blender/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e/can/inference_model_final_wo_optim-889ae475/lm_crop_can_test/results.pkl",
    "output/gdrn/lm_crop_blender/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e/cat/inference_model_final_wo_optim-c754e587/lm_crop_cat_test/results.pkl",
    "output/gdrn/lm_crop_blender/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e/driller/inference_model_final_wo_optim-c1cc7169/lm_crop_driller_test/results.pkl",
    "output/gdrn/lm_crop_blender/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e/duck/inference_model_final_wo_optim-084c3364/lm_crop_duck_test/results.pkl",
    "output/gdrn/lm_crop_blender/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e/holepuncher/inference_model_final_wo_optim-86d52ab7/lm_crop_holepuncher_test/results.pkl",
    "output/gdrn/lm_crop_blender/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e/iron/inference_model_final_wo_optim-6bdb61b0/lm_crop_iron_test/results.pkl",
    "output/gdrn/lm_crop_blender/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e/lamp/inference_model_final_wo_optim-bf52eba3/lm_crop_lamp_test/results.pkl",
    "output/gdrn/lm_crop_blender/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_bboxCrop_DZI10_lm_blender_100e/phone/inference_model_final_wo_optim-4172540b/lm_crop_phone_test/results.pkl",
]

obj_names = list(id2obj.values())
for obj, res_path in zip(obj_names, res_paths):
    cmd = (
        "python core/self6dpp/tools/compute_mIoU_mask.py " "--dataset lm_crop_{}_test --cls {} " "--res_path {}"
    ).format(obj, obj, res_path)
    print(cmd)
    os.system(cmd)
