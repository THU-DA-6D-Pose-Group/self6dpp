import argparse
import os
import os.path as osp
import sys
import numpy as np
from sklearn.metrics import f1_score
from detectron2.data import DatasetCatalog, MetadataCatalog
import mmcv
from tqdm import tqdm

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../../"))
import ref
from core.self6dpp.datasets.dataset_factory import register_datasets
from lib.utils.mask_utils import cocosegm2mask

"""
python core/self6dpp/tools/compute_f1_score_mask.py \
    --dataset lmo_ape_bop_test --cls ape  \
    --res_path output/gdrn/lmoPbrSO/resnest50d_online_AugCosyAAEGray_mlBCE_DoubleMask_lmo_pbr_100e/ape/inference_model_final_wo_optim-f0ef90df/lmo_bop_test/results.pkl \
    --mask_type full_mask

python core/self6dpp/tools/compute_f1_score_mask.py \
    --dataset lmo_can_bop_test --cls can \
    --res_path output/gdrn/lmoPbrSO/resnest50d_online_AugCosyAAEGray_mlBCE_DoubleMask_lmo_pbr_100e/can/inference_model_final_wo_optim-ea5b9c78/lmo_bop_test/results.pkl \
    --mask_type full_mask

python core/self6dpp/tools/compute_f1_score_mask.py \
    --dataset lmo_cat_bop_test --cls cat \
    --res_path output/gdrn/lmoPbrSO/resnest50d_online_AugCosyAAEGray_mlBCE_DoubleMask_lmo_pbr_100e/cat/inference_model_final_wo_optim-9931aeed/lmo_bop_test/results.pkl \
    --mask_type mask

python core/self6dpp/tools/compute_f1_score_mask.py \
    --dataset lmo_driller_bop_test --cls driller \
    --res_path output/gdrn/lmoPbrSO/resnest50d_online_AugCosyAAEGray_mlBCE_DoubleMask_lmo_pbr_100e/driller/inference_model_final_wo_optim-bded40f0/lmo_bop_test/results.pkl \
    --mask_type mask

python core/self6dpp/tools/compute_f1_score_mask.py \
    --dataset lmo_duck_bop_test --cls duck \
    --res_path output/gdrn/lmoPbrSO/resnest50d_online_AugCosyAAEGray_mlBCE_DoubleMask_lmo_pbr_100e/duck/inference_model_final_wo_optim-3cc3dbe6/lmo_bop_test/results.pkl \
    --mask_type mask

python core/self6dpp/tools/compute_f1_score_mask.py \
    --dataset lmo_eggbox_bop_test --cls eggbox \
    --res_path output/gdrn/lmoPbrSO/resnest50d_online_AugCosyAAEGray_mlBCE_DoubleMask_lmo_pbr_100e/eggbox/inference_model_final_wo_optim-817002cd/lmo_bop_test/results.pkl \
    --mask_type mask
    
python core/self6dpp/tools/compute_f1_score_mask.py \
    --dataset lmo_glue_bop_test --cls glue \
    --res_path output/gdrn/lmoPbrSO/resnest50d_online_AugCosyAAEGray_mlBCE_DoubleMask_lmo_pbr_100e/glue/inference_model_final_wo_optim-0b8a2e73/lmo_bop_test/results.pkl \
    --mask_type full_mask

python core/self6dpp/tools/compute_f1_score_mask.py \
    --dataset lmo_holepuncher_bop_test --cls holepuncher \
    --res_path output/gdrn/lmoPbrSO/resnest50d_online_AugCosyAAEGray_mlBCE_DoubleMask_lmo_pbr_100e/holepuncher/inference_model_final_wo_optim-c98281c9/lmo_bop_test/results.pkl \
    --mask_type full_mask         
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Compute f1 score of mask.")
    parser.add_argument("--dataset", type=str, help="dataset_name")
    parser.add_argument("--cls", type=str, help="class name")
    parser.add_argument("--mask_type", type=str, default="mask", help="mask type: mask | full_mask")
    parser.add_argument("--res_path", type=str, help="path to results.")
    args = parser.parse_args()
    return args


args = parse_args()
HEIGHT = 480
WIDTH = 640


def name_long2short(long_name):
    name_split = long_name.split("_")
    short_name = "{:02d}{}".format(int(name_split[0]), "".join([e.capitalize() for e in name_split[1:]]))
    return short_name


def eval_f1_score():
    # load gt mask
    # load pred mask
    # compute score
    dset_name = args.dataset
    res_path = args.res_path
    cls_name = args.cls
    assert osp.exists(res_path), res_path
    assert cls_name in dset_name, f"{cls_name} not in {dset_name}"
    if "ycbv" in dset_name:
        short_name = name_long2short(cls_name)
        assert short_name in res_path, f"{short_name} not in {res_path}"
    else:
        assert cls_name in res_path, f"{cls_name} not in {res_path}"
    print("dataset: ", dset_name)
    print("class: ", cls_name)
    print("results path: ", res_path)
    register_datasets([dset_name])
    dataset_dicts = DatasetCatalog.get(dset_name)
    dset_meta = MetadataCatalog.get(dset_name)
    data_ref = ref.__dict__[dset_meta.ref_key]
    objs = dset_meta.objs
    obj_id = data_ref.obj2id[cls_name]

    # predictions
    init_res = mmcv.load(res_path)

    f1_scores = []
    num_detected = 0
    for dic in tqdm(dataset_dicts):
        scene_im_id = dic["scene_im_id"]
        if scene_im_id not in init_res:
            f1_scores.append(0.0)
            continue

        # load pred mask
        pred_obj_ids = [_r["obj_id"] for _r in init_res[scene_im_id]]
        if obj_id not in pred_obj_ids:
            f1_scores.append(0.0)
            continue

        num_detected += 1
        pred_ind = pred_obj_ids.index(obj_id)

        pred_mask_rle = init_res[scene_im_id][pred_ind][args.mask_type]
        pred_mask = cocosegm2mask(pred_mask_rle, h=HEIGHT, w=WIDTH)
        # load gt mask
        annos = dic["annotations"]
        for anno_i, anno in enumerate(annos):
            obj_name_i = objs[anno["category_id"]]
            if obj_name_i == cls_name:
                if args.mask_type == "mask":
                    gt_rle = anno["segmentation"]
                elif args.mask_type == "full_mask":
                    gt_rle = anno["mask_full"]
                else:
                    raise ValueError("wrong mask type: {}".format(args.mask_type))
                gt_mask = cocosegm2mask(gt_rle, h=HEIGHT, w=WIDTH)
                break
        cur_f1 = f1_score(gt_mask.flatten(), pred_mask.flatten(), average="binary")
        f1_scores.append(cur_f1)

    print("dataset name: ", dset_name, "object: ", cls_name)
    print("dataset length: ", len(dataset_dicts))
    print("length of F1 scores: ", len(f1_scores))
    print("num detected: ", num_detected)
    print(f"{args.mask_type} mean F1 score (%): {np.mean(f1_scores) * 100:.2f}")


if __name__ == "__main__":
    eval_f1_score()
