import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.cnn import normal_init, constant_init
from lib.torch_utils.layers.layer_utils import get_norm, get_nn_act_func
from lib.torch_utils.layers.conv_module import ConvModule


class ConvOutHead(nn.Module):
    def __init__(
        self,
        in_dim,
        num_feat_layers=0,
        feat_dim=256,
        feat_kernel_size=3,
        norm="GN",
        num_gn_groups=32,
        act="GELU",
        out_kernel_size=1,
        out_dim=1,
        num_classes=1,
    ):
        """
        TODO: add feat_strides a list with length num_feat_layers
        """
        super().__init__()
        assert out_kernel_size in [
            1,
            3,
        ], "Only support output kernel size: 1 and 3"

        self.features = nn.ModuleList()
        for i in range(num_feat_layers):
            _in_dim = in_dim if i == 0 else feat_dim
            self.features.append(
                ConvModule(
                    _in_dim,
                    feat_dim,
                    kernel_size=feat_kernel_size,
                    padding=(feat_kernel_size - 1) // 2,
                    norm=norm,
                    num_gn_groups=num_gn_groups,
                    act=act,
                )
            )

        self.num_classes = num_classes
        self.out_dim = out_dim
        _in_dim = feat_dim if num_feat_layers > 0 else in_dim
        self.out_layer = nn.Conv2d(
            _in_dim,
            self.out_dim * num_classes,
            kernel_size=out_kernel_size,
            padding=(out_kernel_size - 1) // 2,
            bias=True,
        )
        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
        normal_init(self.out_layer, std=0.01)

    def forward(self, x):
        for i, l in enumerate(self.features):
            x = l(x)
        out = self.out_layer(x)
        return out
