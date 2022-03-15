"""evaluate coco metric from the detection results (json file) in bop
format."""
import argparse
import os.path as osp
import sys
from tqdm import tqdm
import mmcv
from mmcv import Config
from detectron2.data import DatasetCatalog, MetadataCatalog

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../.."))
from det.yolov4.datasets.dataset_factory import register_datasets
from lib.utils.utils import iprint, dprint
import ref
from det.yolov4.engine.yolov4_coco_evaluation import YOLOv4COCOEvaluator
from lib.utils.setup_logger import setup_my_logger


def get_parser():
    parser = argparse.ArgumentParser(description="Eval coco metric of bop detection results")
    parser.add_argument(
        "--path",
        default="",
        help="path to detection results in bop format (json file)",
    )
    parser.add_argument(
        "--dataset",
        default="lmo_bop_test",
        help="pre-defined dataset name: lm_13_test | lmo_test",
    )
    # optional args
    parser.add_argument("--out_dir", default="output/", help="output dir for evaluation")
    parser.add_argument(
        "--mask_on",
        default=False,
        action="store_true",
        help="evaluate mask or not",
    )
    return parser


def main():
    args = get_parser().parse_args()
    mmcv.mkdir_or_exist(args.out_dir)
    setup_my_logger(name="core")
    setup_my_logger(name="mylib")
    setup_my_logger(name="d2")
    setup_my_logger(name="det")
    setup_my_logger(name="adet")

    cfg = dict(
        MODEL=dict(MASK_ON=args.mask_on, KEYPOINT_ON=False),
        TEST=dict(KEYPOINT_OKS_SIGMAS=[]),
    )
    cfg = Config(cfg)

    dset_name = args.dataset
    dprint("dataset: ", dset_name)
    register_datasets([dset_name])

    meta = MetadataCatalog.get(dset_name)
    objs = meta.objs
    ref_key = meta.ref_key
    data_ref = ref.__dict__[ref_key]

    evaluator = YOLOv4COCOEvaluator(dset_name, cfg, False, output_dir=args.out_dir)

    dicts = DatasetCatalog.get(dset_name)

    dprint("results load from: ", args.path)
    results = mmcv.load(args.path)

    evaluator.reset()
    for scene_im_id, preds in tqdm(results.items()):
        # get gts
        for dic in dicts:
            if dic["scene_im_id"] == scene_im_id:
                image_id = dic["image_id"]
                break

        instances = []
        for pred in preds:
            result = {
                "image_id": image_id,
                "category_id": objs.index(data_ref.id2obj[pred["obj_id"]]),
                "bbox": pred["bbox_est"],  # xywh
                "score": pred["score"],
            }
            # TODO: add mask
            instances.append(result)
        coco_preds = {"image_id": image_id, "instances": instances}
        evaluator._predictions.append(coco_preds)

    # evaluate all images
    dprint("eval results")
    evaluator.evaluate()


if __name__ == "__main__":
    main()
