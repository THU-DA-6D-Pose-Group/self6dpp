import os

id2obj = {
    2: "benchvise",
    7: "driller",
    21: "phone",
}  # bop 2, hb-v1:1  # bop 7, hb-v1:6  # bop 21, hb-v1:20

res_paths = [
    "output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/benchvise/inference_model_final_wo_optim-85b3563e/hb_bdp_benchvise_test_lmK/results.pkl",
    "output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/driller/inference_model_final_wo_optim-4cfc7d64/hb_bdp_driller_test_lmK/results.pkl",
    "output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/phone/inference_model_final_wo_optim-525a29f8/hb_bdp_phone_test_lmK/results.pkl",
]

obj_names = list(id2obj.values())
for obj, res_path in zip(obj_names, res_paths):
    cmd = (
        "python core/self6dpp/tools/compute_mIoU_mask.py " "--dataset hb_bdp_{}_test_lmK --cls {} " "--res_path {}"
    ).format(obj, obj, res_path)
    print(cmd)
    os.system(cmd)
