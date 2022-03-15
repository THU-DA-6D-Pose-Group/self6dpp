from mmcv import Config
import os.path as osp
import os
from tqdm import tqdm

cur_dir = osp.normpath(osp.dirname(osp.abspath(__file__)))

base_cfg_name = "ss_v1_dibr_mlBCE_FreezeBN_woCenter_refinePM10_ape.py"
base_obj_name = "ape"

# -----------------------------------------------------------------
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
obj2id = {_name: _id for _id, _name in id2obj.items()}


init_weights_dict = {
    "ape": "output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/ape/model_final_wo_optim-e8c99c96.pth",
    "benchvise": "output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/benchvise/model_final_wo_optim-85b3563e.pth",
    "camera": "output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/camera/model_final_wo_optim-1b281dbe.pth",
    "can": "output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/can/model_final_wo_optim-53ea56ee.pth",
    "cat": "output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/cat/model_final_wo_optim-f38cfafd.pth",
    "driller": "output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/driller/model_final_wo_optim-4cfc7d64.pth",
    "duck": "output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/duck/model_final_wo_optim-0bde58bb.pth",
    "eggbox": "output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/eggbox_Rsym/model_final_wo_optim-d0656ca7.pth",
    "glue": "output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/glue_Rsym/model_final_wo_optim-324d8f16.pth",
    "holepuncher": "output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/holepuncher/model_final_wo_optim-eab19662.pth",
    "iron": "output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/iron/model_final_wo_optim-025a740e.pth",
    "lamp": "output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/lamp/model_final_wo_optim-34042758.pth",
    "phone": "output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/phone/model_final_wo_optim-525a29f8.pth",
}


def main():
    base_cfg_path = osp.join(cur_dir, base_cfg_name)
    assert osp.exists(base_cfg_path), base_cfg_path  # make sure base cfg is in this dir
    cfg = Config.fromfile(base_cfg_path)

    for obj_id, obj_name in tqdm(id2obj.items()):
        if obj_name in [base_obj_name, "bowl", "cup"]:  # NOTE: ignore base_obj and some unwanted objs
            continue
        print(obj_name)
        # NOTE: what fields should be updated ---------------------------
        new_cfg_dict = dict(
            _base_="./{}".format(base_cfg_name),
            OUTPUT_DIR=cfg.OUTPUT_DIR.replace(base_obj_name, obj_name),
            DATASETS=dict(
                TRAIN=("lm_real_{}_train".format(obj_name),),  # real data
                TRAIN2=("lm_pbr_{}_train".format(obj_name),),  # synthetic data
                TRAIN2_RATIO=0.0,
                TEST=("lm_real_{}_test".format(obj_name),),
            ),
            MODEL=dict(
                # synthetically trained model
                WEIGHTS=init_weights_dict[obj_name]
            ),
        )
        # ----------------------------------------------------------------------
        new_cfg_path = osp.join(cur_dir, base_cfg_name.replace(base_obj_name, obj_name))
        if osp.exists(new_cfg_path):
            raise RuntimeError("new cfg exists!")
        new_cfg = Config(new_cfg_dict)
        with open(new_cfg_path, "w") as f:
            f.write(new_cfg.pretty_text)

    # re-format
    os.system("black -l 120 {}".format(cur_dir))


if __name__ == "__main__":
    main()
