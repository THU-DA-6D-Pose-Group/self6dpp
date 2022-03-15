# -*- coding: utf-8 -*-
import copy
import logging
import os.path as osp
import pickle

import cv2
import mmcv
import numpy as np
import ref
import torch

from core.base_data_loader import Base_DatasetFromList
from lib.structures import Center2Ds, Keypoints2Ds, Keypoints3Ds, MyBitMasks, Translations, Poses, MyList
from core.utils.data_utils import read_image_mmcv
from core.utils.dataset_utils import (
    filter_empty_dets,
    filter_invalid_in_dataset_dicts,
    flat_dataset_dicts,
    load_detections_into_dataset,
    load_init_poses_into_dataset,
    my_build_batch_data_loader,
    trivial_batch_collator,
)
from core.utils.pose_aug import aug_poses_normal_np
from core.utils.pose_utils import rot_from_axangle_chain
from core.utils.my_distributed_sampler import InferenceSampler, RepeatFactorTrainingSampler, TrainingSampler
from core.utils.ssd_color_transform import ColorAugSSDTransform
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import get_detection_dataset_dicts
from detectron2.data import transforms as T
from detectron2.structures import Boxes, BoxMode, Instances, Keypoints, PolygonMasks
from detectron2.utils.logger import log_first_n
from lib.pysixd import inout, misc
from lib.utils.mask_utils import cocosegm2mask, get_edge

from .dataset_factory import register_datasets

logger = logging.getLogger(__name__)


def build_deepim_augmentation(cfg, is_train):
    """Create a list of :class:`Augmentation` from config. when training 6d
    pose, cannot flip.

    Returns:
        list[Augmentation]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    augmentation = []
    augmentation.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    if is_train:
        # augmentation.append(T.RandomFlip())
        logger.info("Augmentations used in training: " + str(augmentation))
    return augmentation


def transform_instance_annotations(
    annotation,
    transforms,
    image_size,
    *,
    keypoint_hflip_indices=None,
    bbox_key="bbox",
):
    """
    NOTE: Adapted from detection_utils.
    Apply transforms to box, segmentation, keypoints, etc. of annotations of a single instance.

    It will use `transforms.apply_box` for the box, and
    `transforms.apply_coords` for segmentation polygons & keypoints.
    If you need anything more specially designed for each data structure,
    you'll need to implement your own version of this function or the transforms.

    Args:
        annotation (dict): dict of instance annotations for a single instance.
        transforms (TransformList):
        image_size (tuple): the height, width of the transformed image
        keypoint_hflip_indices (ndarray[int]): see `create_keypoint_hflip_indices`.

    Returns:
        dict:
            the same input dict with fields `bbox_key`, "segmentation", "keypoints"
            transformed according to `transforms`.
            The "bbox_mode" field will be set to XYXY_ABS.
    """
    im_H, im_W = image_size
    bbox = BoxMode.convert(annotation[bbox_key], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    # Note that bbox is 1d (per-instance bounding box)
    annotation[bbox_key] = np.array(transforms.apply_box([bbox])[0])
    annotation["bbox_mode"] = BoxMode.XYXY_ABS

    if "segmentation" in annotation:
        # NOTE: here we transform segms to binary masks (interp is nearest by default)
        if isinstance(annotation["segmentation"], np.ndarray):
            mask_ori = annotation["segmentation"]
        else:
            mask_ori = cocosegm2mask(annotation["segmentation"], h=im_H, w=im_W)
        mask = transforms.apply_segmentation(mask_ori)
        annotation["segmentation"] = mask  # NOTE: visib_mask
        if "trunc_mask" not in annotation:
            annotation["trunc_mask"] = mask.copy()

    # TODO: maybe also load obj_masks (full masks)
    if "trunc_mask" in annotation:
        annotation["trunc_mask"] = transforms.apply_segmentation(annotation["trunc_mask"])

    if "keypoints" in annotation:
        keypoints = utils.transform_keypoint_annotations(
            annotation["keypoints"],
            transforms,
            image_size,
            keypoint_hflip_indices,
        )
        annotation["keypoints"] = keypoints

    if "centroid_2d" in annotation:
        annotation["centroid_2d"] = transforms.apply_coords(np.array(annotation["centroid_2d"]).reshape(1, 2)).flatten()

    return annotation


def annotations_to_instances(cfg, annos, image_size, mask_format="bitmask", K=None):
    """# NOTE: modified from detection_utils Create an :class:`Instances`
    object used by the models, from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields:
                "obj_classes",
                "obj_poses",
                "obj_boxes",
                "obj_boxes_det",  (used for keep original detected bbox xyxy)
                "obj_masks",
                "obj_3d_points"
            if they can be obtained from `annos`.
    """
    boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
    insts = Instances(image_size)  # (h,w)
    boxes = insts.obj_boxes = Boxes(boxes)
    boxes.clip(image_size)

    if all("bbox_det" in obj for obj in annos):
        boxes_det = [BoxMode.convert(obj["bbox_det"], obj["bbox_det_mode"], BoxMode.XYXY_ABS) for obj in annos]
        insts.obj_boxes_det = Boxes(boxes_det)
        insts.obj_boxes_det.clip(image_size)

    classes = [obj["category_id"] for obj in annos]
    # some properties do not need a new structure
    classes = torch.tensor(classes, dtype=torch.int64)
    insts.obj_classes = classes

    if len(annos) and "segmentation" in annos[0]:
        segms = [obj["segmentation"] for obj in annos]
        if mask_format == "polygon":
            masks = PolygonMasks(segms)
        else:
            assert mask_format == "bitmask", mask_format
            # NOTE: should be kept on cpu, otherwise CUDA out of memory?
            # masks = BitMasks(torch.stack([torch.from_numpy(x) for x in segms]))
            masks = MyBitMasks(torch.stack([torch.tensor(x.copy()) for x in segms]), cpu_only=True)

        insts.obj_visib_masks = masks

    if len(annos) and "trunc_mask" in annos[0]:
        segms = [obj["trunc_mask"] for obj in annos]
        # NOTE: should be kept on cpu, otherwise CUDA out of memory?
        masks = MyBitMasks(torch.stack([torch.tensor(x.copy()) for x in segms]), cpu_only=True)
        insts.obj_trunc_masks = masks

    # NOTE: pose related annotations
    # for train: this is gt pose
    # for test: this is init pose
    if len(annos) and "pose" in annos[0]:
        poses = [obj["pose"] for obj in annos]
        insts.obj_poses = Poses(poses)

    if len(annos) and "score" in annos[0]:  # for test
        scores = [obj["score"] for obj in annos]
        insts.obj_scores = torch.tensor(scores, dtype=torch.float32)

    return insts


def filter_empty_instances(instances, by_box=True, by_mask=True, box_threshold=1e-5):
    """Filter out empty instances in an `Instances` object.

    Args:
        instances (Instances):
        by_box (bool): whether to filter out instances with empty boxes
        by_mask (bool): whether to filter out instances with empty masks
        box_threshold (float): minimum width and height to be considered non-empty

    Returns:
        Instances: the filtered instances.
    """
    assert by_box or by_mask
    r = []
    if by_box:
        r.append(instances.obj_boxes.nonempty(threshold=box_threshold))
    if instances.has("obj_visib_masks") and by_mask:
        r.append(instances.obj_visib_masks.nonempty())

    # TODO: can also filter visible keypoints

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x  # bool tensor

    return instances[m]


class DeepIM_DatasetFromList(Base_DatasetFromList):
    """NOTE: we can also use the default DatasetFromList and
    implement a similar custom DataMapper,
    but it is harder to implement some features relying on other dataset dicts
    # https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/common.py
    Wrap a list to a torch Dataset. It produces elements of the list as data.
    """

    def __init__(self, cfg, split, lst: list, copy: bool = True, serialize: bool = True, flatten=False):
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
        self.augmentation = build_deepim_augmentation(cfg, is_train=(split == "train"))
        if cfg.INPUT.COLOR_AUG_PROB > 0 and cfg.INPUT.COLOR_AUG_TYPE.lower() == "ssd":
            self.augmentation.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
            logging.getLogger(__name__).info("Color augmentation used in training: " + str(self.augmentation[-1]))
        # fmt: off
        self.img_format = cfg.INPUT.FORMAT  # default BGR
        self.with_depth = cfg.INPUT.WITH_DEPTH
        self.aug_depth = cfg.INPUT.AUG_DEPTH
        # NOTE: color augmentation config
        self.color_aug_prob = cfg.INPUT.COLOR_AUG_PROB
        self.color_aug_type = cfg.INPUT.COLOR_AUG_TYPE
        self.color_aug_code = cfg.INPUT.COLOR_AUG_CODE
        # fmt: on
        self.cfg = cfg
        self.split = split  # train | val | test
        if split == "train" and self.color_aug_prob > 0:
            self.color_augmentor = self._get_color_augmentor(aug_type=self.color_aug_type, aug_code=self.color_aug_code)
        else:
            self.color_augmentor = None
        # ------------------------
        # common model infos
        self.fps_points = {}
        self.model_points = {}
        self.extents = {}
        self.sym_infos = {}
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

    def _get_fps_points(self, dataset_name, with_center=False):
        """convert to label based keys.

        # TODO: get models info similarly
        """
        if dataset_name in self.fps_points:
            return self.fps_points[dataset_name]

        dset_meta = MetadataCatalog.get(dataset_name)
        ref_key = dset_meta.ref_key
        data_ref = ref.__dict__[ref_key]
        objs = dset_meta.objs
        cfg = self.cfg
        num_fps_points = cfg.MODEL.DEEPIM.XYZ_HEAD.NUM_REGIONS  # TODO: maybe add a xyz head
        cur_fps_points = {}
        loaded_fps_points = data_ref.get_fps_points()
        for i, obj_name in enumerate(objs):
            obj_id = data_ref.obj2id[obj_name]
            if with_center:
                cur_fps_points[i] = loaded_fps_points[str(obj_id)][f"fps{num_fps_points}_and_center"]
            else:
                cur_fps_points[i] = loaded_fps_points[str(obj_id)][f"fps{num_fps_points}_and_center"][:-1]
        self.fps_points[dataset_name] = cur_fps_points
        return self.fps_points[dataset_name]

    def _get_model_points(self, dataset_name):
        """convert to label based keys."""
        if dataset_name in self.model_points:
            return self.model_points[dataset_name]

        dset_meta = MetadataCatalog.get(dataset_name)
        ref_key = dset_meta.ref_key
        data_ref = ref.__dict__[ref_key]
        objs = dset_meta.objs
        cfg = self.cfg

        cur_model_points = {}
        num = np.inf
        for i, obj_name in enumerate(objs):
            obj_id = data_ref.obj2id[obj_name]
            model_path = osp.join(data_ref.model_dir, f"obj_{obj_id:06d}.ply")
            model = inout.load_ply(model_path, vertex_scale=data_ref.vertex_scale)
            cur_model_points[i] = pts = model["pts"]
            if pts.shape[0] < num:
                num = pts.shape[0]

        num = min(num, cfg.MODEL.DEEPIM.LOSS_CFG.NUM_PM_POINTS)
        for i in range(len(cur_model_points)):
            keep_idx = np.arange(num)
            np.random.shuffle(keep_idx)  # random sampling
            cur_model_points[i] = cur_model_points[i][keep_idx, :]

        self.model_points[dataset_name] = cur_model_points
        return self.model_points[dataset_name]

    def _get_extents(self, dataset_name):
        """label based keys."""
        if dataset_name in self.extents:
            return self.extents[dataset_name]

        dset_meta = MetadataCatalog.get(dataset_name)
        try:
            ref_key = dset_meta.ref_key
        except:
            # FIXME: for some reason, in distributed training, this need to be re-registered
            register_datasets([dataset_name])
            dset_meta = MetadataCatalog.get(dataset_name)
            ref_key = dset_meta.ref_key

        data_ref = ref.__dict__[ref_key]
        objs = dset_meta.objs
        cfg = self.cfg

        cur_extents = {}
        for i, obj_name in enumerate(objs):
            obj_id = data_ref.obj2id[obj_name]
            model_path = osp.join(data_ref.model_dir, f"obj_{obj_id:06d}.ply")
            model = inout.load_ply(model_path, vertex_scale=data_ref.vertex_scale)
            pts = model["pts"]
            xmin, xmax = np.amin(pts[:, 0]), np.amax(pts[:, 0])
            ymin, ymax = np.amin(pts[:, 1]), np.amax(pts[:, 1])
            zmin, zmax = np.amin(pts[:, 2]), np.amax(pts[:, 2])
            size_x = xmax - xmin
            size_y = ymax - ymin
            size_z = zmax - zmin
            cur_extents[i] = np.array([size_x, size_y, size_z], dtype="float32")

        self.extents[dataset_name] = cur_extents
        return self.extents[dataset_name]

    def _get_sym_infos(self, dataset_name):
        """label based keys."""
        if dataset_name in self.sym_infos:
            return self.sym_infos[dataset_name]

        dset_meta = MetadataCatalog.get(dataset_name)
        ref_key = dset_meta.ref_key
        data_ref = ref.__dict__[ref_key]
        objs = dset_meta.objs
        cfg = self.cfg

        cur_sym_infos = {}
        loaded_models_info = data_ref.get_models_info()
        for i, obj_name in enumerate(objs):
            obj_id = data_ref.obj2id[obj_name]
            model_info = loaded_models_info[str(obj_id)]
            if "symmetries_discrete" in model_info or "symmetries_continuous" in model_info:
                sym_transforms = misc.get_symmetry_transformations(model_info, max_sym_disc_step=0.01)
                sym_info = np.array([sym["R"] for sym in sym_transforms], dtype=np.float32)
            else:
                sym_info = None
            cur_sym_infos[i] = sym_info

        self.sym_infos[dataset_name] = cur_sym_infos
        return self.sym_infos[dataset_name]

    def read_data(self, dataset_dict):
        """load image and annos random shift & scale bbox; crop, rescale."""
        cfg = self.cfg
        net_cfg = cfg.MODEL.DEEPIM
        pose_head_cfg = net_cfg.POSE_HEAD
        mask_head_cfg = net_cfg.MASK_HEAD

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        dataset_name = dataset_dict["dataset_name"]

        image = read_image_mmcv(dataset_dict["file_name"], format=self.img_format)
        # should be consistent with the size in dataset_dict
        utils.check_image_size(dataset_dict, image)
        im_H_ori, im_W_ori = image.shape[:2]

        # maybe replace bg and maybe truncate the fg for each object ===============
        # currently only replace bg for train ###############################
        if self.split == "train":
            # some synthetic data already has bg, img_type should be real or something else but not syn
            img_type = dataset_dict.get("img_type", "real")
            _replace_bg = False
            if img_type == "syn":
                log_first_n(logging.WARNING, "replace bg", n=10)
                _replace_bg = True
            else:  # real image
                if np.random.rand() < cfg.INPUT.CHANGE_BG_PROB:
                    log_first_n(logging.WARNING, "replace bg for real", n=10)
                    _replace_bg = True

            if _replace_bg:
                assert "annotations" in dataset_dict and "segmentation" in dataset_dict["annotations"][0]
                _trunc_fg = cfg.INPUT.get("TRUNCATE_FG", False)
                for anno_i, anno in enumerate(dataset_dict["annotations"]):
                    anno["segmentation"] = visib_mask = cocosegm2mask(anno["segmentation"], im_H_ori, im_W_ori)
                    if _trunc_fg:
                        anno["trunc_mask"] = self.trunc_mask(visib_mask).astype("uint8")

                if _trunc_fg:
                    trunc_masks = [anno["trunc_mask"] for anno in dataset_dict["annotations"]]
                    fg_mask = sum(trunc_masks).astype("bool").astype("uint8")
                else:
                    visib_masks = [anno["segmentation"] for anno in dataset_dict["annotations"]]
                    fg_mask = sum(visib_masks).astype("bool").astype("uint8")
                image = self.replace_bg(image.copy(), fg_mask, return_mask=False, truncate_fg=False)
        ######## replace bg done #############################################################################

        # NOTE: color augment here ===================================
        if self.split == "train" and self.color_aug_prob > 0 and self.color_augmentor is not None:
            if np.random.rand() < self.color_aug_prob:
                if cfg.INPUT.COLOR_AUG_SYN_ONLY and img_type not in ["real"]:
                    image = self._color_aug(image, self.color_aug_type)
                else:
                    image = self._color_aug(image, self.color_aug_type)

        # other transforms (mainly geometric ones) ---------------------------------
        # for 6d pose task, flip is not allowed in general except for some 2d keypoints methods
        image, transforms = T.apply_augmentations(self.augmentation, image)
        im_H, im_W = image_shape = image.shape[:2]  # h, w

        # NOTE: scale camera intrinsic if necessary ================================
        scale_x = im_W / im_W_ori
        scale_y = im_H / im_H_ori  # NOTE: generally scale_x should be equal to scale_y
        if "cam" in dataset_dict:
            if im_W != im_W_ori or im_H != im_H_ori:
                dataset_dict["cam"][0] *= scale_x
                dataset_dict["cam"][1] *= scale_y
            K = dataset_dict["cam"].astype("float32")
            dataset_dict["cam"] = torch.as_tensor(K)

        # image (normalized)-----------------------------------------
        image = self.normalize_image(cfg, image.transpose(2, 0, 1))
        # CHW, float32 tensor
        dataset_dict["image"] = torch.as_tensor(image.astype("float32")).contiguous()

        # CHW -> HWC
        # coord_2d = get_2d_coord_np(im_W, im_H, low=0, high=1).transpose(1, 2, 0)

        ## for test ############################################################################
        if self.split != "train":
            # determine test box and init pose type---------------------------------------------
            bbox_key, pose_key = get_test_bbox_initpose_key(cfg)
            # ---------------------------------------
            # "annotations" means detections
            test_annos = dataset_dict["annotations"]
            obj_extents = []
            obj_points = []
            obj_sym_infos = []
            for inst_i, test_anno in enumerate(test_annos):
                obj_cls = test_anno["category_id"]
                obj_extents.append(self._get_extents(dataset_name)[obj_cls])  # extent
                obj_points.append(self._get_model_points(dataset_name)[obj_cls])  # (V,3)
                obj_sym_infos.append(self._get_sym_infos(dataset_name)[obj_cls])  # sym_info
                # get test bbox, init pose(maybe None)  # keys become: bbox, [pose]
                self._get_test_bbox_initpose(
                    test_anno, bbox_key, pose_key, K=K, dataset_name=dataset_name, imW=im_W, imH=im_H
                )

            test_annos = [
                transform_instance_annotations(_anno, transforms, image_shape)
                for _anno in dataset_dict.pop("annotations")
            ]
            # construct test instances ===============================
            # obj_classes, obj_boxes, [obj_poses] [obj_boxes_det]
            test_insts = annotations_to_instances(cfg, test_annos, image_shape)
            # obj_extents, obj_sym_infos --------------------------------------------
            test_insts.obj_extents = torch.tensor(np.array(obj_extents, dtype="float32"))
            test_insts.obj_points = torch.tensor(np.array(obj_points).astype("float32"))
            test_insts.obj_sym_infos = MyList(obj_sym_infos)
            dataset_dict["instances"] = test_insts
            return dataset_dict

        ## for train ####################################################################################
        if "annotations" in dataset_dict:
            # transform annotations ------------------------
            # NOTE: newly added keys should consider whether they need to be transformed
            annos = [
                transform_instance_annotations(obj_anno, transforms, image_shape)
                for obj_anno in dataset_dict.pop("annotations")
            ]

            # augment bboxes ===================================================
            # NOTE: be careful to not influence other annotations
            for anno in annos:
                if np.random.rand() < cfg.INPUT.BBOX_AUG_PROB:  # apply prob instance-wise
                    anno["bbox"] = self.aug_bbox_non_square(cfg, anno["bbox"], im_H, im_W)

            # construct instances ===============================
            # obj_classes, obj_boxes, obj_poses, obj_visib_masks, obj_trunc_masks
            instances = annotations_to_instances(cfg, annos, image_shape)

            # obj_extents, obj_points, obj_sym_infos ----------------------------------------
            obj_extents = []
            obj_points = []
            obj_sym_infos = []
            for obj_cls in instances.obj_classes:
                obj_cls = int(obj_cls)
                obj_extents.append(self._get_extents(dataset_name)[obj_cls])  # (3,)
                obj_points.append(self._get_model_points(dataset_name)[obj_cls])  # (V,3)
                obj_sym_infos.append(self._get_sym_infos(dataset_name)[obj_cls])
            instances.obj_extents = torch.tensor(np.array(obj_extents).astype("float32"))
            instances.obj_points = torch.tensor(np.array(obj_points).astype("float32"))
            instances.obj_sym_infos = MyList(obj_sym_infos)

            # instances infos -------------------------------
            # NOTE: if no instance left after filtering, will return un-filtered one
            dataset_dict["instances"] = filter_empty_instances(instances, by_box=True, by_mask=True)

        return dataset_dict

    def _get_test_bbox_initpose(self, test_anno, bbox_key, pose_key, K, dataset_name, imW=640, imH=480):
        cfg = self.cfg
        # get init poses (or None) -------------------------------------------------
        if pose_key == "pose_gt_noise":
            # in this case, we load the gt poses/annotations and add random noise
            gt_pose = test_anno["pose"]
            pose_gt_noise = aug_poses_normal_np(
                gt_pose[None],
                std_rot=cfg.INPUT.NOISE_ROT_STD_TEST,
                std_trans=cfg.INPUT.NOISE_TRANS_STD_TEST,
                rot_max=cfg.INPUT.NOISE_ROT_MAX_TEST,
                min_z=cfg.INPUT.INIT_TRANS_MIN_Z,
            )[0]
            test_anno["pose"] = pose_gt_noise
        elif pose_key == "pose_est":
            test_anno["pose"] = test_anno.pop("pose_est")
        elif pose_key == "pose_canonical":
            rot = rot_from_axangle_chain(cfg.INPUT.CANONICAL_ROT)
            trans = np.array(cfg.INPUT.CANONICAL_TRANS)
            test_anno["pose"] = np.hstack([rot, trans.reshape(3, 1)]).astype("float32")
        else:
            raise ValueError("Unknown test init_pose type: {}".format(pose_key))

        # get test boxes ------------------------------------------------------
        if bbox_key != "bbox_est" and "bbox_est" in test_anno:
            test_anno["bbox_det"] = test_anno["bbox_est"]
            test_anno["bbox_det_mode"] = test_anno["bbox_mode"]

        if bbox_key == "bbox_est":
            test_anno["bbox"] = test_anno.pop("bbox_est")
            test_anno["bbox_det"] = test_anno["bbox"]
            test_anno["bbox_det_mode"] = test_anno["bbox_mode"]
        elif bbox_key == "bbox_pose":  # compute box from pose
            assert (
                pose_key != "pose_canonical"
            ), "Compute test bbox from canonical pose may not contain the object in image!"
            obj_cls = test_anno["category_id"]
            points = self._get_model_points(dataset_name)[obj_cls]
            test_anno["bbox"] = misc.compute_2d_bbox_xyxy_from_pose(
                points, test_anno["pose"], K, width=imW, height=imH, clip=True
            )
            test_anno["bbox_mode"] = BoxMode.XYXY_ABS
        elif bbox_key == "bbox_gt_aug":
            bbox_gt = BoxMode.convert(test_anno.pop("bbox"), test_anno["bbox_mode"], BoxMode.XYXY_ABS)
            test_anno["bbox"] = self.aug_bbox_non_square(cfg, bbox_gt, im_H=imH, im_W=imW)
            test_anno["bbox_mode"] = BoxMode.XYXY_ABS
        else:  # gt
            if "bbox" not in test_anno:
                raise RuntimeError("No gt bbox for test!")
        # inplace modification, do not return

    def __getitem__(self, idx):
        if self.split != "train":
            dataset_dict = self._get_sample_dict(idx)
            return self.read_data(dataset_dict)

        while True:  # return valid data for train
            dataset_dict = self._get_sample_dict(idx)
            processed_data = self.read_data(dataset_dict)
            if processed_data is None:
                idx = self._rand_another(idx)
                continue
            return processed_data


def build_deepim_train_loader(cfg, dataset_names):
    """A data loader is created by the following steps:

    1. Use the dataset names in config to query :class:`DatasetCatalog`, and obtain a list of dicts.
    2. Coordinate a random shuffle order shared among all processes (all GPUs)
    3. Each process spawn another few workers to process the dicts. Each worker will:
       * Map each metadata dict into another format to be consumed by the model.
       * Batch them by simply putting dicts into a list.

    The batched ``list[mapped_dict]`` is what this dataloader will yield.

    Args:
        cfg: the config

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

    dataset = DeepIM_DatasetFromList(cfg, split="train", lst=dataset_dicts, copy=False, flatten=False)

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


def build_deepim_test_loader(cfg, dataset_name, train_objs=None):
    """Similar to `build_detection_train_loader`. But this function uses the
    given `dataset_name` argument (instead of the names in cfg), and uses batch
    size 1.

    Args:
        cfg:
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

    # load test detection results -------------------------
    if cfg.MODEL.LOAD_DETS_TEST:
        det_files = cfg.DATASETS.DET_FILES_TEST
        assert len(cfg.DATASETS.TEST) == len(det_files)
        dataset_dicts = load_detections_into_dataset(
            dataset_name,
            dataset_dicts,
            det_file=det_files[cfg.DATASETS.TEST.index(dataset_name)],
            top_k_per_obj=cfg.DATASETS.DET_TOPK_PER_OBJ,
            score_thr=cfg.DATASETS.DET_THR,
            train_objs=train_objs,
        )
        if cfg.DATALOADER.FILTER_EMPTY_DETS:
            dataset_dicts = filter_empty_dets(dataset_dicts)

    # load test init poses ------------------------------------
    if cfg.MODEL.LOAD_POSES_TEST:
        if cfg.MODEL.LOAD_DETS_TEST:
            logger.warning("Override loaded test bboxes by loading test init poses.")
        init_pose_files = cfg.DATASETS.INIT_POSE_FILES_TEST
        assert len(init_pose_files) == len(cfg.DATASETS.TEST)
        load_init_poses_into_dataset(
            dataset_name,
            dataset_dicts,
            init_pose_file=init_pose_files[cfg.DATASETS.TEST.index(dataset_name)],
            top_k_per_obj=cfg.DATASETS.INIT_POSE_TOPK_PER_OBJ,
            score_thr=cfg.DATASETS.INIT_POSE_THR,
            train_objs=train_objs,
        )
        if cfg.DATALOADER.FILTER_EMPTY_DETS:
            dataset_dicts = filter_empty_dets(dataset_dicts)

    dataset = DeepIM_DatasetFromList(cfg, split="test", lst=dataset_dicts, flatten=False)

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
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
        **kwargs,
    )
    return data_loader


def get_test_bbox_initpose_key(cfg):
    test_bbox_type = cfg.INPUT.BBOX_TYPE_TEST
    test_pose_init_type = cfg.INPUT.INIT_POSE_TYPE_TEST  # gt_noise | est | canonical
    bbox_initpose_types_to_keys = {
        "est/est": ("bbox_est", "pose_est"),  # common test case 1
        "from_pose/est": (
            "bbox_pose",
            "pose_est",
        ),  # common test case 2, compute bbox from pose and 3D points
        "est/canonical": (
            "bbox_est",
            "pose_canonical",
        ),  # common test case 3 (also predict initial pose)
        # theses are only for validation
        "gt/canonical": ("bbox", "pose_canonical"),
        "gt_aug/canonical": ("bbox_gt_aug", "pose_canonical"),
        "est/gt_noise": ("bbox_est", "pose_gt_noise"),
        "from_pose/gt_noise": ("bbox_pose", "pose_gt_noise"),
    }
    return bbox_initpose_types_to_keys[f"{test_bbox_type}/{test_pose_init_type}"]
