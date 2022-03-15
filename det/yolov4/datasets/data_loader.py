# -*- coding: utf-8 -*-
import copy
import logging
import pickle

import mmcv
import random
import numpy as np
import torch
from core.base_data_loader import Base_DatasetFromList
from core.utils.data_utils import read_image_mmcv
from core.utils.dataset_utils import (
    filter_invalid_in_dataset_dicts,
    flat_dataset_dicts,
    my_build_batch_data_loader,
    trivial_batch_collator,
)
from core.utils.my_distributed_sampler import (
    InferenceSampler,
    RepeatFactorTrainingSampler,
    TrainingSampler,
)
from detectron2.data import get_detection_dataset_dicts
from detectron2.structures import BoxMode

from .datasets_misc import letterbox, random_affine, augment_hsv
from det.yolov4.yolo_utils.utils import xyxy2cxcywh

logger = logging.getLogger(__name__)


class YoloV4_DatasetFromList(Base_DatasetFromList):
    """NOTE: we can also use the default DatasetFromList and
    implement a similar custom DataMapper,
    but it is harder to implement some features relying on other dataset dicts
    # https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/common.py
    Wrap a list to a torch Dataset. It produces elements of the list as data.
    """

    def __init__(
        self, cfg, split, lst: list, *, stride, img_size, copy: bool = True, serialize: bool = True, flatten=False
    ):
        """
        Args:
            lst (list): a list which contains elements to produce.
            copy (bool): whether to deepcopy the element when producing it,
                so that the result can be modified in place without affecting the
                source in the list.
            serialize (bool): whether to hold memory using serialized objects, when
                enabled, data loader workers can use shared RAM from master
                process instead of making a copy.
        """
        self.cfg = cfg
        self.split = split  # train | val | test
        # fmt: off
        self.stride = stride
        self.img_size = img_size
        # load 4 images at a time into a mosaic (only during training)
        self.mosaic = cfg.INPUT.AUG_MOSAIC if split == 'train' else False
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.img_format = cfg.INPUT.FORMAT  # default BGR
        self.with_depth = cfg.INPUT.WITH_DEPTH
        self.aug_depth = cfg.INPUT.AUG_DEPTH
        # NOTE: color augmentation config
        self.aug_hsv_prob = cfg.INPUT.AUG_HSV_PROB
        self.color_aug_prob = cfg.INPUT.COLOR_AUG_PROB
        self.color_aug_type = cfg.INPUT.COLOR_AUG_TYPE
        self.color_aug_code = cfg.INPUT.COLOR_AUG_CODE
        self.rand_hflip = cfg.INPUT.RAND_HFLIP
        self.rand_vflip = cfg.INPUT.RAND_VFLIP
        # fmt: on
        if split == "train" and self.color_aug_prob > 0:
            self.color_augmentor = self._get_color_augmentor(aug_type=self.color_aug_type, aug_code=self.color_aug_code)
        else:
            self.color_augmentor = None
        # ----------------------------------------------------
        self.flatten = flatten
        self._lst = flat_dataset_dicts(lst) if flatten else lst
        # ----------------------------------------------------
        self._copy = copy
        self._serialize = serialize

        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        if self._serialize:
            logger.info("Serializing {} elements to byte tensors and concatenating them all ...".format(len(self._lst)))
            self._lst = [_serialize(x) for x in self._lst]
            self._addr = np.asarray([len(x) for x in self._lst], dtype=np.int64)
            self._addr = np.cumsum(self._addr)
            self._lst = np.concatenate(self._lst)
            logger.info("Serialized dataset takes {:.2f} MiB".format(len(self._lst) / 1024 ** 2))

    def __len__(self):
        if self._serialize:
            return len(self._addr)
        else:
            return len(self._lst)

    def read_data(self, idx):
        """load image and annos random shift & scale bbox; crop, rescale."""
        cfg = self.cfg
        net_cfg = cfg.MODEL.YOLO
        loss_cfg = net_cfg.LOSS_CFG

        dataset_dict = copy.deepcopy(self._get_sample_dict(idx))  # it will be modified by code below

        if self.split == "train":
            if self.mosaic:
                image, labels = self.load_mosaic(idx)
                shapes = None
            else:
                (
                    image,
                    (im_H_ori, im_W_ori),
                    (im_H, im_W),
                ) = self.load_resize_image(dataset_dict)
                # Letterbox
                shape = self.img_size  # final letterboxed shape
                image, ratio, pad = letterbox(image, shape, auto=False, scaleup=True)
                shapes = (im_H_ori, im_W_ori), (
                    (im_W / im_W_ori, im_W / im_W_ori),
                    pad,
                )  # for COCO mAP rescaling

                # Load labels
                annotations = dataset_dict["annotations"]
                bboxes = [BoxMode.convert(anno["bbox"], anno["bbox_mode"], BoxMode.XYXY_ABS) for anno in annotations]
                bboxes = np.array(bboxes, dtype="float32")
                classes = np.array([anno["category_id"] for anno in annotations])
                labels = np.hstack([classes.reshape(-1, 1), bboxes])
                if labels.size > 0:
                    labels[:, 1] = ratio[0] * im_W / im_W_ori * labels[:, 1] + pad[0]  # pad width
                    labels[:, 2] = ratio[1] * im_H / im_H_ori * labels[:, 2] + pad[1]  # pad height
                    labels[:, 3] = ratio[0] * im_W / im_W_ori * labels[:, 3] + pad[0]
                    labels[:, 4] = ratio[1] * im_H / im_H_ori * labels[:, 4] + pad[1]

                image, labels = random_affine(
                    image,
                    labels,
                    degrees=cfg.INPUT.RAND_ROTATE_DEG,
                    translate=cfg.INPUT.RAND_TRANSLATE,
                    scale=cfg.INPUT.RAND_SCALE,
                    shear=cfg.INPUT.RAND_SHEAR,
                )
        else:  # load test image
            image, (im_H_ori, im_W_ori), (im_H, im_W) = self.load_resize_image(dataset_dict)
            # Letterbox
            shape = self.img_size  # final letterboxed shape
            image, ratio, pad = letterbox(image, shape, auto=False, scaleup=False)
            shapes = (im_H_ori, im_W_ori), (
                (im_W / im_W_ori, im_W / im_W_ori),
                pad,
            )  # for COCO mAP rescaling

        if self.split == "train":
            # Augment colorspace
            if np.random.rand() < self.aug_hsv_prob:
                augment_hsv(
                    image,
                    hgain=cfg.INPUT.HSV_H,
                    sgain=cfg.INPUT.HSV_S,
                    vgain=cfg.INPUT.HSV_V,
                    source_format=self.img_format,
                )

        # NOTE: maybe add or change color augment here ===================================
        if self.split == "train" and self.color_aug_prob > 0 and self.color_augmentor is not None:
            if np.random.rand() < self.color_aug_prob:
                image = self._color_aug(image, self.color_aug_type)

        if self.split == "train":
            # convert xyxy to cxcywh (rel) --------------------------
            n_label = len(labels)
            if n_label > 0:
                labels[:, 1:5] = xyxy2cxcywh(labels[:, 1:5])
                # Normalize coordinates 0 - 1
                labels[:, [2, 4]] /= image.shape[0]  # height
                labels[:, [1, 3]] /= image.shape[1]  # width

            # random left-right flip
            if self.rand_hflip and random.random() < 0.5:
                image = np.fliplr(image)
                if n_label > 0:
                    labels[:, 1] = 1 - labels[:, 1]  # flip cx

            # random up-down flip
            if self.rand_vflip and random.random() < 0.5:
                image = np.flipud(image)
                if n_label > 0:
                    labels[:, 2] = 1 - labels[:, 2]  # flip cy

        dataset_dict.pop("annotations", None)  # no need to keep original annos

        # result image: NOTE: yolo was trained in RGB for coco (CHW, 0-1)
        if self.img_format == "BGR":
            image_normed = self.normalize_image(cfg, image[:, :, ::-1].transpose(2, 0, 1))
        elif self.img_format == "RGB":
            image_normed = self.normalize_image(cfg, image.transpose(2, 0, 1))
        else:
            raise ValueError(
                "Yolo was trained in RGB. In dataloader, RGB or BGR are OK, but got: {}".format(self.img_format)
            )
        dataset_dict["image"] = torch.as_tensor(image_normed.astype("float32")).contiguous()
        dataset_dict["shapes"] = shapes

        #################################################################################
        if self.split != "train":
            # don't load annotations at test time
            return dataset_dict
        #######################################################################################
        labels_out = torch.zeros(n_label, 6)  # image_idx_in_batch, cls_label, cxcywh
        if n_label > 0:
            labels_out[:, 1:] = torch.from_numpy(labels)
        dataset_dict["labels"] = labels_out
        return dataset_dict

    def load_resize_image(self, dataset_dict):
        # loads 1 image from dataset, returns img, original hw, resized hw
        img = read_image_mmcv(dataset_dict["file_name"], format=self.img_format)  # BGR
        assert img is not None, "Image not found: {}".format(dataset_dict["file_name"])
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = "area" if r < 1 and self.split != "train" else "bilinear"
            img = mmcv.imresize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized

    def load_mosaic(self, idx):
        # loads images in a mosaic
        labels4 = []
        im_size = self.img_size
        yc, xc = [int(random.uniform(-x, 2 * im_size + x)) for x in self.mosaic_border]  # mosaic center x, y
        indices = [idx] + [random.randint(0, self.__len__() - 1) for _ in range(3)]  # 3 additional image indices

        for i, index in enumerate(indices):
            # Load image
            dataset_dict = self._get_sample_dict(idx)
            img, (h0, w0), (h, w) = self.load_resize_image(dataset_dict)

            # place img in img4
            if i == 0:  # top left
                img4 = np.full(
                    (im_size * 2, im_size * 2, img.shape[2]),
                    114,
                    dtype=np.uint8,
                )  # base image with 4 tiles
                x1a, y1a, x2a, y2a = (
                    max(xc - w, 0),
                    max(yc - h, 0),
                    xc,
                    yc,
                )  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = (
                    w - (x2a - x1a),
                    h - (y2a - y1a),
                    w,
                    h,
                )  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = (
                    xc,
                    max(yc - h, 0),
                    min(xc + w, im_size * 2),
                    yc,
                )
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = (
                    max(xc - w, 0),
                    yc,
                    xc,
                    min(im_size * 2, yc + h),
                )
                x1b, y1b, x2b, y2b = (
                    w - (x2a - x1a),
                    0,
                    max(xc, w),
                    min(y2a - y1a, h),
                )
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = (
                    xc,
                    yc,
                    min(xc + w, im_size * 2),
                    min(im_size * 2, yc + h),
                )
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            annotations = dataset_dict["annotations"]
            bboxes = [BoxMode.convert(anno["bbox"], anno["bbox_mode"], BoxMode.XYXY_ABS) for anno in annotations]
            bboxes = np.array(bboxes, dtype="float32")
            classes = np.array([anno["category_id"] for anno in annotations])
            labels = np.hstack([classes.reshape(-1, 1), bboxes])
            if labels.size > 0:  # old xyxy to new xyxy
                labels[:, 1] = w / w0 * labels[:, 1] + padw
                labels[:, 2] = h / h0 * labels[:, 2] + padh
                labels[:, 3] = w / w0 * labels[:, 3] + padw
                labels[:, 4] = h / h0 * labels[:, 4] + padh
            labels4.append(labels)

        # Concat/clip labels
        if len(labels4):
            labels4 = np.concatenate(labels4, 0)
            # np.clip(labels4[:, 1:] - im_size / 2, 0, im_size, out=labels4[:, 1:])  # use with center crop
            np.clip(labels4[:, 1:], 0, 2 * im_size, out=labels4[:, 1:])  # use with random_affine

            # Replicate
            # img4, labels4 = replicate(img4, labels4)

        # Augment
        # img4 = img4[s // 2: int(s * 1.5), s // 2:int(s * 1.5)]  # center crop (WARNING, requires box pruning)
        cfg = self.cfg
        img4, labels4 = random_affine(
            img4,
            labels4,
            degrees=cfg.INPUT.RAND_ROTATE_DEG,
            translate=cfg.INPUT.RAND_TRANSLATE,
            scale=cfg.INPUT.RAND_SCALE,
            shear=cfg.INPUT.RAND_SHEAR,
            border=self.mosaic_border,
        )  # border to remove

        return img4, labels4

    def __getitem__(self, idx):
        if self.split != "train":
            return self.read_data(idx)

        while True:  # return valid data for train
            processed_data = self.read_data(idx)
            if processed_data is None:
                idx = self._rand_another(idx)
                continue
            return processed_data


def build_yolo_train_loader(cfg, dataset_names, *, stride, img_size):
    """A data loader is created by the following steps:

    1. Use the dataset names in config to query :class:`DatasetCatalog`, and obtain a list of dicts.
    2. Coordinate a random shuffle order shared among all processes (all GPUs)
    3. Each process spawn another few workers to process the dicts. Each worker will:
       * Map each metadata dict into another format to be consumed by the model.
       * Batch them by simply putting dicts into a list.

    The batched ``list[mapped_dict]`` is what this dataloader will yield.

    Args:
        cfg (CfgNode): the config

    Returns:
        an infinite iterator of training data
    """
    dataset_dicts = get_detection_dataset_dicts(
        dataset_names,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE if cfg.MODEL.KEYPOINT_ON else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
    )

    dataset_dicts = filter_invalid_in_dataset_dicts(dataset_dicts, visib_thr=cfg.DATALOADER.FILTER_VISIB_THR)

    dataset = YoloV4_DatasetFromList(
        cfg,
        split="train",
        lst=dataset_dicts,
        stride=stride,
        img_size=img_size,
        copy=False,
    )

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))
    # TODO avoid if-else?
    if sampler_name == "TrainingSampler":
        sampler = TrainingSampler(len(dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
            dataset_dicts, cfg.DATALOADER.REPEAT_THRESHOLD
        )
        sampler = RepeatFactorTrainingSampler(repeat_factors)
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))
    return my_build_batch_data_loader(
        dataset,
        sampler,
        cfg.SOLVER.IMS_PER_BATCH,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )


def build_yolo_test_loader(cfg, dataset_name, *, stride, img_size):
    """Similar to `build_detection_train_loader`. But this function uses the
    given `dataset_name` argument (instead of the names in cfg), and uses batch
    size 1.

    Args:
        cfg: a detectron2 CfgNode
        dataset_name (str): a name of the dataset that's available in the DatasetCatalog

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.
    """
    dataset_dicts = get_detection_dataset_dicts(
        [dataset_name],
        filter_empty=False,
        proposal_files=[cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(dataset_name)]]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )

    dataset = YoloV4_DatasetFromList(
        cfg,
        split="test",
        lst=dataset_dicts,
        stride=stride,
        img_size=img_size,
        flatten=False,
    )

    sampler = InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    # Horovod: limit # of CPU threads to be used per worker.
    # if num_workers > 0:
    #     torch.set_num_threads(num_workers)

    kwargs = {"num_workers": num_workers}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    # https://github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_imagenet_resnet50.py
    # if (num_workers > 0 and hasattr(mp, '_supports_context') and
    #         mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
    #     kwargs['multiprocessing_context'] = 'forkserver'
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=batch_sampler, collate_fn=trivial_batch_collator, **kwargs
    )
    return data_loader
