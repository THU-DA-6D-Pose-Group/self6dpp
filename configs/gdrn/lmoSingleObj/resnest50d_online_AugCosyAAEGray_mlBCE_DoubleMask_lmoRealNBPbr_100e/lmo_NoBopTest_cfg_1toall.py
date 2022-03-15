from mmcv import Config
import os.path as osp
import os
from tqdm import tqdm

cur_dir = osp.normpath(osp.dirname(osp.abspath(__file__)))

base_cfg_name = "resnest50d_online_AugCosyAAEGray_mlBCE_DoubleMask_lmoRealNBPbr_100e_01_ape_NoBopTest.py"
base_id_obj = "01_ape"
base_obj_name = "ape"

# -----------------------------------------------------------------
id2obj = {
    1: "ape",
    5: "can",
    6: "cat",
    8: "driller",
    9: "duck",
    10: "eggbox",
    11: "glue",
    12: "holepuncher",
}
obj2id = {_name: _id for _id, _name in id2obj.items()}


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
                TRAIN=(
                    "lmo_pbr_{}_train".format(obj_name),
                    "lmo_NoBopTest_{}_train".format(obj_name),
                )
            ),
        )
        # ----------------------------------------------------------------------
        new_cfg_path = osp.join(cur_dir, base_cfg_name.replace(base_id_obj, f"{obj_id:02d}_{obj_name}"))
        if osp.exists(new_cfg_path):
            raise RuntimeError("new cfg exists!")
        new_cfg = Config(new_cfg_dict)
        with open(new_cfg_path, "w") as f:
            f.write(new_cfg.pretty_text)

    # re-format
    os.system("black -l 120 {}".format(cur_dir))


if __name__ == "__main__":
    main()
