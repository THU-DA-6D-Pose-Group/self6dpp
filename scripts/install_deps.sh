#!/usr/bin/env bash
# some other dependencies
set -x
install=${1:-"all"}

if test "$install" = "all"; then
echo "Installing apt dependencies"
sudo apt-get install -y libjpeg-dev zlib1g-dev
sudo apt-get install -y libopenexr-dev
sudo apt-get install -y openexr
sudo apt-get install -y python3-dev
sudo apt-get install -y libglfw3-dev libglfw3
sudo apt-get install -y libglew-dev
sudo apt-get install -y libassimp-dev
sudo apt-get install -y libnuma-dev  # for byteps
sudo apt install -y clang
## for bop cpp renderer
sudo apt install -y curl
sudo apt install -y autoconf
sudo apt-get install -y build-essential libtool

## for uncertainty pnp
sudo apt-get install -y libeigen3-dev
sudo apt-get install -y libgoogle-glog-dev
sudo apt-get install -y libsuitesparse-dev
sudo apt-get install -y libatlas-base-dev

## for nvdiffrast/egl
sudo apt-get install -y --no-install-recommends \
    cmake curl pkg-config
sudo apt-get install -y --no-install-recommends \
    libgles2 \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev
# (only available for Ubuntu >= 18.04)
sudo apt-get install -y --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libglvnd-dev

sudo apt-get install -y libglew-dev
# for GLEW, add this into ~/.bashrc
# export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
fi

# conda install ipython
pip install cython
pip install plyfile

pip install pycocotools  # or install the nvidia version which is cpp-accelerated
# git clone https://github.com/NVIDIA/cocoapi.git cocoapi_nvidia
# cd cocoapi_nvidia/PythonAPI
# make
# python setup.py build develop

pip install cffi
pip install ninja
pip install black
pip install docformatter
pip install setproctitle
pip install fastfunc
pip install meshplex
pip install OpenEXR
pip install "vispy>=0.6.4"
pip install tabulate
pip install pytest-runner
pip install pytest
pip install ipdb
pip install tqdm
pip install numba
pip install mmcv-full
pip install imagecorruptions
pip install pyassimp==4.1.3  # 4.1.4 will cause egl_renderer SegmentFault
pip install pypng
pip install "imgaug>=0.4.0"
pip install albumentations
pip install transforms3d
# pip install pyquaternion
pip install torchvision
pip install open3d
pip install fvcore
pip install tensorboardX
pip install einops
pip install pytorch3d
pip install timm  # pytorch-image-models
pip install glfw
pip install imageio imageio-ffmpeg
pip install PyOpenGL PyOpenGL_accelerate  # >=3.1.5
pip install chardet

pip install thop  # https://github.com/Lyken17/pytorch-OpCounter
pip install loguru

# verified versions
onnx==1.8.1
onnxruntime==1.8.0
onnx-simplifier==0.3.5

# pip install kornia

pip uninstall pillow
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd

# install kaolin

# install detectron2
# git clone https://github.com/facebookresearch/detectron2.git
# cd detectron2 && pip install -e .

# install adet  # https://github.com/aim-uofa/adet.git
# git clone https://github.com/aim-uofa/adet.git
# cd adet
# python setup.py build develop
