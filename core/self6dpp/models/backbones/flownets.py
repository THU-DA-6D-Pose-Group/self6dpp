"""modified from
https://github.com/NVIDIA/flownet2-pytorch/blob/master/networks/FlowNetS.py."""
from __future__ import print_function, absolute_import, division
import os.path as osp
import logging

cur_dir = osp.dirname(osp.abspath(__file__))
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import _load_checkpoint, load_state_dict


logger = logging.getLogger(__name__)


class FlowNetS(nn.Module):
    """Parameter count : > 38,676,504"""

    def __init__(
        self,
        in_channels=6,
        use_bn=False,
        out_flow_level="flow4",
        out_concat4=True,
    ):
        """
        Args:
            out_flow_level: flow4 (default) | all ([flow2, flow3, flow4, flow5, flow6]) | none
        """
        super().__init__()
        self.in_channels = in_channels
        self.use_bn = use_bn
        self.out_flow_level = out_flow_level
        self.out_concat4 = out_concat4

        # layers
        self.conv1 = conv(self.use_bn, self.in_channels, 64, kernel_size=7, stride=2)  # scale 1/2
        self.conv2 = conv(self.use_bn, 64, 128, kernel_size=5, stride=2)  # scale 1/4
        self.conv3 = conv(self.use_bn, 128, 256, kernel_size=5, stride=2)  # scale 1/8
        self.conv3_1 = conv(self.use_bn, 256, 256, kernel_size=3, stride=1)
        self.conv4 = conv(self.use_bn, 256, 512, kernel_size=3, stride=2)  # scale 1/16
        self.conv4_1 = conv(self.use_bn, 512, 512, kernel_size=3, stride=1)
        self.conv5 = conv(self.use_bn, 512, 512, kernel_size=3, stride=2)  # scale 1/32
        self.conv5_1 = conv(self.use_bn, 512, 512, kernel_size=3, stride=1)
        self.conv6 = conv(self.use_bn, 512, 1024, kernel_size=3, stride=2)  # scale 1/64
        self.conv6_1 = conv(self.use_bn, 1024, 1024, kernel_size=3, stride=1)

        # flow and deconv+conv feat
        if self.out_flow_level != "none" or self.out_concat4:
            self.predict_flow6 = predict_flow(1024)  # conv3x3 with 2 out channels
            self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
            self.deconv5 = deconv(1024, 512)

            self.predict_flow5 = predict_flow(1026)
            self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
            self.deconv4 = deconv(1026, 256)

            if self.out_flow_level != "none":
                self.predict_flow4 = predict_flow(770)
                if self.out_flow_level == "all":
                    self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
                    self.deconv3 = deconv(770, 128)

                    self.predict_flow3 = predict_flow(386)
                    self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
                    self.deconv2 = deconv(386, 64)

                    self.predict_flow2 = predict_flow(194)
        # NOTE: init it outside manually
        # self.init_weights(pretrained=pretrained)

    def _init_from_pretrained(self, pretrained):
        checkpoint = _load_checkpoint(pretrained, map_location="cpu")
        # OrderedDict is a subclass of dict
        if not isinstance(checkpoint, dict):
            raise RuntimeError(f"No state_dict found in checkpoint file {pretrained}")
        # get state_dict from checkpoint
        if "state_dict" in checkpoint:
            pretrained_dict = checkpoint["state_dict"]
        else:
            pretrained_dict = checkpoint

        my_state_dict = self.state_dict()
        for name, v in pretrained_dict.items():
            if name in my_state_dict.keys():
                if name == "conv1.0.weight":
                    my_state_dict[name][:, 0:6, :, :] = v
                else:
                    my_state_dict[name] = v
        load_state_dict(self, my_state_dict, logger=logger)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str) and len(pretrained) > 0:
            self._init_from_pretrained(pretrained=pretrained)
        else:
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    kaiming_init(m, bias=0, distribution="normal")
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1, bias=0)

    def get_out_conv_feat_dim(self):
        return 1024

    def get_out_deconv_feat_dim(self):
        if self.out_flow_level != "none" or self.out_concat4:
            return 770
        else:
            return 0

    def forward(self, x):
        """
        Args:
            x: (,in_channels,H,W)
        Returns:
            out_conv6: (,1024,H/64,W/64)
            concat4: (,770,H/16,W/16) or None
            flow: None | flow4 | [flow2, flow3, flow4, flow5, flow6]
        """
        out_conv1 = self.conv1(x)  # (, 64, 240,320)
        out_conv2 = self.conv2(out_conv1)  # (, 128, 120,160)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))  # (, 256, 60, 80)
        out_conv4 = self.conv4_1(self.conv4(out_conv3))  # (, 512, 30, 40)
        out_conv5 = self.conv5_1(self.conv5(out_conv4))  # (, 512, 15, 20)
        out_conv6 = self.conv6_1(self.conv6(out_conv5))  # (, 1024, 8, 10)

        if self.out_flow_level != "none" or self.out_concat4:
            # optical flow
            flow6 = self.predict_flow6(out_conv6)  # (, 2, 8, 10)
            flow6_up = self.upsampled_flow6_to_5(flow6)  # (, 2, 16, 20)
            crop_flow6_up = crop_like(flow6_up, out_conv5)  # crop the upper left (, 2, 15, 20)
            out_deconv5 = self.deconv5(out_conv6)  # (, 512, 16, 20)
            crop_out_deconv5 = crop_like(out_deconv5, out_conv5)  # (, 512, 15, 20)

            concat5 = torch.cat((out_conv5, crop_out_deconv5, crop_flow6_up), 1)  # (, 1026, 15, 20)
            flow5 = self.predict_flow5(concat5)  # (,2,15,20)
            flow5_up = self.upsampled_flow5_to_4(flow5)  # (,2,30,40)
            crop_flow5_up = crop_like(flow5_up, out_conv4)  # (,2,30,40)
            out_deconv4 = self.deconv4(concat5)  # (,256,30,40)
            crop_out_deconv4 = crop_like(out_deconv4, out_conv4)  # (,256, 30, 40)

            concat4 = torch.cat((out_conv4, crop_out_deconv4, crop_flow5_up), 1)  # (,770,30,40)
            # NOTE: mx-deepim use concat4 as deconv feat and then predict flow/mask
            if self.out_flow_level != "none":
                # here we use multi-level flow predicts to keep it the same as original flownet
                flow4 = self.predict_flow4(concat4)  # (,2,30,40)
                if self.out_flow_level == "all":
                    flow4_up = self.upsampled_flow4_to_3(flow4)  # (,2,60,80)
                    crop_flow4_up = crop_like(flow4_up, out_conv3)  # (,2,60,80)
                    out_deconv3 = self.deconv3(concat4)  # (,128,60,80)
                    crop_out_deconv3 = crop_like(out_deconv3, out_conv3)  # (,128,60,80)

                    concat3 = torch.cat((out_conv3, crop_out_deconv3, crop_flow4_up), 1)  # (,386,60,80)
                    flow3 = self.predict_flow3(concat3)  # (,2,60,80)
                    flow3_up = self.upsampled_flow3_to_2(flow3)  # (,2,120,160)
                    crop_flow3_up = crop_like(flow3_up, out_conv2)  # (,2,120,160)
                    out_deconv2 = self.deconv2(concat3)  # (,64,120,160)
                    crop_out_deconv2 = crop_like(out_deconv2, out_conv2)  # (,64,120,160)

                    concat2 = torch.cat((out_conv2, crop_out_deconv2, crop_flow3_up), 1)  # (,194,120,160)
                    flow2 = self.predict_flow2(concat2)  # (,2,120,160)
                    flow_out = [flow2, flow3, flow4, flow5, flow6]
                else:
                    flow_out = flow4
            else:
                flow_out = None
        else:
            concat4 = None
            flow_out = None
        # output: conv, deconv, flow
        return out_conv6, concat4, flow_out

    @property
    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if "weight" in name]

    @property
    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if "bias" in name]


def conv(use_bn, in_planes, out_planes, kernel_size=3, stride=1):
    if use_bn:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                bias=True,
            ),
            nn.LeakyReLU(0.1, inplace=True),
        )


def crop_like(input, target):
    """crop the input like target in the upper left.

    another way is to use F.pad(flow6_up, (0, 0, -1, 0))  # eg: crop the
    up side, (,2,16,20) --> (, 2, 15, 20)
    """
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, : target.size(2), : target.size(3)]


def deconv(in_planes, out_planes, bias=True):
    # in nvidia, bias is True; in deepim, bias is False
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_planes,
            out_planes,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=bias,
        ),
        nn.LeakyReLU(0.1, inplace=True),
    )


def predict_flow(in_planes, bias=True):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=bias)


if __name__ == "__main__":
    device = "cuda"
    pretrained_path = osp.join(
        cur_dir,
        "../../../../pretrained_models/flownet/flownets_EPE1.951.pth.tar",
    )
    model = FlowNetS(in_channels=6, use_bn=False, out_flow_level="flow4", out_concat4=True)
    model.init_weights(pretrained=pretrained_path)
    model.to(device)

    # print('*'*20)
    print(model.state_dict().keys())
    print("model device: ", next(model.parameters()).device)
