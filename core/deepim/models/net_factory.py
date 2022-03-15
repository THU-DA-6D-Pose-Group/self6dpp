from .backbones.flownets import FlowNetS
from .heads.fc_rot_trans_head import FC_RotTransHead
from .heads.conv_out_head import ConvOutHead
from .heads.top_down_head import TopDownHead
import timm


BACKBONES = {"FlowNetS": FlowNetS}

# register timm models
# yapf: disable
# "resnet18", "resnet18d", "tv_resnet34", "resnet34", "resnet34d", "resnet50", "resnet50d", "resnet101", "resnet101d",
# "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
# "hrnet_w18_small", "hrnet_w18_small_v2", "hrnet_w18", "hrnet_w30", "hrnet_w32", "hrnet_w40", "hrnet_w44", "hrnet_w48", "hrnet_w64",
# only get the models with pretrained models
for backbone_name in timm.list_models(pretrained=True):
    BACKBONES[f"timm/{backbone_name}"] = timm.create_model
# yapf: enable


HEADS = {
    "FC_RotTransHead": FC_RotTransHead,
    "ConvOutHead": ConvOutHead,
    "TopDownHead": TopDownHead,
}
