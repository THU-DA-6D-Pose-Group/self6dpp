import timm

# mmlab models
from .backbones.mm_nets.resnet import ResNet, ResNetV1c
from .backbones.mm_nets.mmcls_resnet import ResNetV1d
from .backbones.mm_nets.resnest import ResNeSt
from .backbones.mm_nets.darknet import Darknet

# torchvision resnet
from .backbones.resnet_backbone import get_resnet

# pvnet resnet_8s
from .backbones.pvnet_net.model_repository import (
    Resnet18_8s,
    Resnet34_8s,
    Resnet50_8s,
    Resnet50_8s_2o,
)

# flownet-s
from .backbones.flownets import FlowNetS

from .necks.fpn import FPN
from .heads.fpn_mask_xyz_region_head import FPNMaskXyzRegionHead
from .heads.top_down_mask_xyz_region_head import TopDownMaskXyzRegionHead
from .heads.top_down_doublemask_xyz_region_head import (
    TopDownDoubleMaskXyzRegionHead,
)
from .heads.conv_mask_xyz_region_head import ConvMaskXyzRegionHead
from .heads.conv_pnp_net import ConvPnPNet
from .heads.conv_pnp_net_no_region import ConvPnPNetNoRegion
from .heads.conv_pnp_net_cls import ConvPnPNetCls
from .heads.point_pnp_net import SimplePointPnPNet

# heads for refiner
from .heads.fc_rot_trans_head import FC_RotTransHead
from .heads.conv_out_head import ConvOutHead


BACKBONES = {
    # pvnet models
    "Resnet18_8s": Resnet18_8s,
    "Resnet34_8s": Resnet34_8s,
    "Resnet50_8s": Resnet50_8s,
    "Resnet50_8s_2o": Resnet50_8s_2o,
    # mmlab models
    "mm/ResNet": ResNet,
    "mm/ResNetV1c": ResNetV1c,
    "mm/ResNetV1d": ResNetV1d,
    "mm/ResNeSt": ResNeSt,
    "mm/Darknet": Darknet,
    # flownet-s
    "FlowNetS": FlowNetS,
}


# register torchvision resnet models
for backbone_name in [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
]:
    BACKBONES[f"tv/{backbone_name}"] = get_resnet


# register timm models
# yapf: disable
# "resnet18", "resnet18d", "tv_resnet34", "resnet34", "resnet34d", "resnet50", "resnet50d", "resnet101", "resnet101d",
# "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
# "hrnet_w18_small", "hrnet_w18_small_v2", "hrnet_w18", "hrnet_w30", "hrnet_w32", "hrnet_w40", "hrnet_w44", "hrnet_w48", "hrnet_w64",
# only get the models with pretrained models
for backbone_name in timm.list_models(pretrained=True):
    BACKBONES[f"timm/{backbone_name}"] = timm.create_model
# yapf: enable

# -------------------------------------------------------------------------------
NECKS = {"FPN": FPN}

# -------------------------------------------------------------------------------
HEADS = {
    # mask-xyz-region
    "TopDownDoubleMaskXyzRegionHead": TopDownDoubleMaskXyzRegionHead,
    "TopDownMaskXyzRegionHead": TopDownMaskXyzRegionHead,
    "ConvMaskXyzRegionHead": ConvMaskXyzRegionHead,
    "FPNMaskXyzRegionHead": FPNMaskXyzRegionHead,
    # pnp net
    "ConvPnPNet": ConvPnPNet,
    "ConvPnPNetNoRegion": ConvPnPNetNoRegion,
    "ConvPnPNetCls": ConvPnPNetCls,
    "SimplePointPnPNet": SimplePointPnPNet,
    # heads for refiner
    "ConvOutHead": ConvOutHead,
    "FC_RotTransHead": FC_RotTransHead,
}
