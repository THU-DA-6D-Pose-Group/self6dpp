import torch
import numpy as np
from lib.vis_utils.image import grid_show, vis_image_bboxes_cv2
from det.yolov4.yolo_utils.utils import cxcywh2xyxy


def batch_data(cfg, data, device="cuda", phase="train"):
    if phase != "train":
        return batch_data_test(cfg, data, device=device)

    # batch training data
    batch = {}
    batch["image"] = torch.stack([d["image"] for d in data], dim=0).to(device, non_blocking=True)
    labels = [d["labels"] for d in data]
    for i, l in enumerate(labels):
        l[:, 0] = i  # add target image index in batch for build_targets()
    batch["labels"] = torch.cat(labels, 0).to(device, non_blocking=True)
    batch["shapes"] = [d["shapes"] for d in data]
    return batch


def batch_data_test(cfg, data, device="cuda"):
    batch = {}
    batch["image"] = torch.stack([d["image"] for d in data], dim=0).to(device, non_blocking=True)
    batch["shapes"] = [d["shapes"] for d in data]
    return batch


def vis_batch(cfg, batch, phase="train"):
    bs = batch["image"].shape[0]
    if phase == "train":
        labels = batch["labels"].cpu().numpy()
    # yapf: disable
    for i in range(bs):
        image = (batch['image'][i].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype('uint8')
        vis_dict = {"image": image[:, :, ::-1]}
        if phase == "train":
            labels_i = [labels[j, 1:] for j in range(len(labels)) if labels[j, 0] == i]
            labels_i = np.array(labels_i)
            cls_labels = labels_i[:, 0]
            boxes_cxcywh = labels_i[:, 1:5]
            boxes_xyxy = cxcywh2xyxy(boxes_cxcywh)
            boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]] * image.shape[0]
            boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]] * image.shape[1]
            text_labels = [str(l) for l in cls_labels]
            im_vis = vis_image_bboxes_cv2(image, boxes_xyxy, labels=text_labels)
            vis_dict["im_vis"] = im_vis[:, :, ::-1]
        show_titles = list(vis_dict.keys())
        show_ims = list(vis_dict.values())
        ncol = 2
        nrow = int(np.ceil(len(show_ims) / ncol))
        grid_show(show_ims, show_titles, row=nrow, col=ncol)
    # yapf: enable
