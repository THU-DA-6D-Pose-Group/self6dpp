# Self6D++

This repo provides the PyTorch implementation of the work:

**Gu Wang &dagger;, Fabian Manhardt &dagger;, Xingyu Liu, Xiangyang Ji &#9993;, Federico Tombari. Occlusion-Aware Self-Supervised Monocular 6D Object Pose Estimation. IEEE Transactions on Pattern Analysis and Machine Intelligence.**
[[Paper](https://doi.org/10.1109/TPAMI.2021.3136301)][[arXiv](https://arxiv.org/abs/2203.10339)][[bibtex](#Citation)]

## Requirements
* Ubuntu 16.04/18.04, CUDA 10.2, python >= 3.6, PyTorch >= 1.7.1, torchvision
* Install `detectron2` from [source](https://github.com/facebookresearch/detectron2)
* `sh scripts/install_deps.sh`
* Compile the cpp extension for `farthest points sampling (fps)`, `optical flow`, `chamfer distance`, and `egl renderer`:
    ```
    sh scripts/compile_all.sh
    ```

## Datasets
Download the 6D pose datasets (LINEMOD, Occluded LINEMOD, YCB-Video) from the
[BOP website](https://bop.felk.cvut.cz/datasets/) and
[VOC 2012](https://pjreddie.com/projects/pascal-voc-dataset-mirror/)
for background images.

The structure of `datasets` folder should look like below:
```
# recommend using soft links (ln -sf)
datasets/
├── BOP_DATASETS   # https://bop.felk.cvut.cz/datasets/
    ├──lm
    ├──lmo
    ├──ycbv
├── lm_renders_blender  # the blender rendered images
├── VOCdevkit
```

## Train and test
* Our method contains two-stages:
    * Stage I: train the detector, pose estimator, and refiner using PBR synthetic data
    * Stage II: self-supervised training for the pose estimator

* In general, for each part, the training and test commands follow the template:
```<train/test_script.sh> <config_path> <gpu_ids> (other args)```
    * `<config_path>` can be found at the directory `configs/`.
    * `<gpu_ids>` can be `0` or `1` for single-gpu training, or `0,1` for multi-gpu training. We use single-gpu training for all the experiments.

* The trained models can be found at [Pan.Baidu, pw: g1x6](https://pan.baidu.com/s/189T6j5OVFXiV7mbik441gQ?pwd=g1x6
), or [Onedrive](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/liuxy21_mails_tsinghua_edu_cn/Eo_1M56pMMJHnIZa6M-VAqYB0z_01MIXrpXl6O2tjPQ4qA?e=uKYNkC).

* Some other resources (including `test_bboxes` and `image_sets`) can be found at [Pan.Baidu, pw: 8nWC](https://pan.baidu.com/s/1lfVXryDPVv3ujCQzUETdjg?pwd=8nWC), or  [Onedrive](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/liuxy21_mails_tsinghua_edu_cn/Eo_1M56pMMJHnIZa6M-VAqYB0z_01MIXrpXl6O2tjPQ4qA?e=uKYNkC).


### Stage I: train the detector, pose estimator, and refiner using PBR synthetic data
#### Train and test Yolov4:
```
det/yolov4/train_yolov4.sh <config_path> <gpu_ids> (other args)

det/yolov4/test_yolov4.sh <config_path> <gpu_ids> (other args)
```

#### Train and test GDR-Net:
```
core/gdrn_modeling/train_gdrn.sh <config_path> <gpu_ids> (other args)

core/gdrn_modeling/test_gdrn.sh <config_path> <gpu_ids> (other args)
```

#### Train and test Refiner (DeepIM):
```
core/deepim/train_deepim.sh <config_path> <gpu_ids> (other args)

core/deepim/test_deepim.sh <config_path> <gpu_ids> (other args)
```


### Stage II: self-supervised training for the pose estimator
#### Train and test Self6D++:
```
core/self6dpp/train_self6dpp.sh <config_path> <gpu_ids> (other args)

core/self6dpp/test_self6dpp.sh <config_path> <gpu_ids> (other args)
```


## Citation
If you find this useful in your research, please consider citing:
```
@article{Wang_2021_self6dpp,
  title     = {Occlusion-Aware Self-Supervised Monocular {6D} Object Pose Estimation},
  author    = {Wang, Gu and Manhardt, Fabian and Liu, Xingyu and Ji, Xiangyang and Tombari, Federico},
  journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  year      = {2021},
  doi       = {10.1109/TPAMI.2021.3136301}
}
```
