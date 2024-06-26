# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved. 
#   
# Licensed under the Apache License, Version 2.0 (the "License");   
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at   
#   
#     http://www.apache.org/licenses/LICENSE-2.0    
# 
# Unless required by applicable law or agreed to in writing, software   
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and   
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F

from picodet.core.workspace import register, serializable
from ..shape_spec import ShapeSpec

__all__ = [
    'CSPDarkNet',
    'BaseConv',
    'DWConv',
    'BottleNeck',
    'SPPLayer',
    'SPPFLayer',
]


def get_activation(name="silu"):
    if name == "silu":
        module = nn.SiLU()
    elif name == "relu":
        module = nn.ReLU()
    elif name in ["LeakyReLU", 'leakyrelu', 'lrelu']:
        module = nn.LeakyReLU(0.1)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return x * F.sigmoid(x)


class BaseConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 groups=1,
                 bias=False,
                 act="silu"):
        super(BaseConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=(ksize - 1) // 2,
            groups=groups,
            bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, track_running_stats=True)

        self._init_weights()

    def _init_weights(self):
        # conv_init_(self.conv)
        pass

    def forward(self, x):
        # use 'x * F.sigmoid(x)' replace 'silu'
        x = self.bn(self.conv(x))
        y = x * F.sigmoid(x)
        return y


class DWConv(nn.Module):
    """Depthwise Conv"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride=1,
                 bias=False,
                 act="silu"):
        super(DWConv, self).__init__()
        self.dw_conv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            bias=bias,
            act=act)
        self.pw_conv = BaseConv(
            in_channels,
            out_channels,
            ksize=1,
            stride=1,
            groups=1,
            bias=bias,
            act=act)

    def forward(self, x):
        return self.pw_conv(self.dw_conv(x))


class Focus(nn.Module):
    """Focus width and height information into channel space, used in YOLOX."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize=3,
                 stride=1,
                 bias=False,
                 act="silu"):
        super(Focus, self).__init__()
        self.conv = BaseConv(
            in_channels * 4,
            out_channels,
            ksize=ksize,
            stride=stride,
            bias=bias,
            act=act)

    def forward(self, inputs):
        # inputs [bs, C, H, W] -> outputs [bs, 4C, W/2, H/2]
        top_left = inputs[:, :, 0::2, 0::2]
        top_right = inputs[:, :, 0::2, 1::2]
        bottom_left = inputs[:, :, 1::2, 0::2]
        bottom_right = inputs[:, :, 1::2, 1::2]
        outputs = torch.concat(
            [top_left, bottom_left, top_right, bottom_right], 1)
        return self.conv(outputs)


class BottleNeck(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 shortcut=True,
                 kernel_sizes=(1, 3),
                 expansion=0.5,
                 depthwise=False,
                 bias=False,
                 act="silu"):
        super(BottleNeck, self).__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(
            in_channels, hidden_channels, ksize=kernel_sizes[0], stride=1, bias=bias, act=act)
        self.conv2 = Conv(
            hidden_channels,
            out_channels,
            ksize=kernel_sizes[1],
            stride=1,
            bias=bias,
            act=act)
        self.add_shortcut = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.add_shortcut:
            y = y + x
        return y


class SPPLayer(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer used in YOLOv3-SPP and YOLOX"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes=(5, 9, 13),
                 bias=False,
                 act="silu"):
        super(SPPLayer, self).__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        self.maxpoolings = nn.ModuleList([
            nn.MaxPool2d(
                kernel_size=ks, stride=1, padding=ks // 2)
            for ks in kernel_sizes
        ])
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(
            conv2_channels, out_channels, ksize=1, stride=1, bias=bias, act=act)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.concat([x] + [mp(x) for mp in self.maxpoolings], dim=1)
        x = self.conv2(x)
        return x


class SPPFLayer(nn.Module):
    """ Spatial Pyramid Pooling - Fast (SPPF) layer used in YOLOv5 by Glenn Jocher,
        equivalent to SPP(k=(5, 9, 13))
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize=5,
                 bias=False,
                 act='silu'):
        super(SPPFLayer, self).__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        self.maxpooling = nn.MaxPool2d(
            kernel_size=ksize, stride=1, padding=ksize // 2)
        conv2_channels = hidden_channels * 4
        self.conv2 = BaseConv(
            conv2_channels, out_channels, ksize=1, stride=1, bias=bias, act=act)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.maxpooling(x)
        y2 = self.maxpooling(y1)
        y3 = self.maxpooling(y2)
        concats = torch.concat([x, y1, y2, y3], dim=1)
        out = self.conv2(concats)
        return out


class CSPLayer(nn.Module):
    """CSP (Cross Stage Partial) layer with 3 convs, named C3 in YOLOv5"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=1,
                 shortcut=True,
                 expansion=0.5,
                 depthwise=False,
                 bias=False,
                 act="silu"):
        super(CSPLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        self.conv2 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            BottleNeck(
                hidden_channels,
                hidden_channels,
                shortcut=shortcut,
                expansion=1.0,
                depthwise=depthwise,
                bias=bias,
                act=act) for _ in range(num_blocks)
        ])
        self.conv3 = BaseConv(
            hidden_channels * 2,
            out_channels,
            ksize=1,
            stride=1,
            bias=bias,
            act=act)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        x = torch.concat([x_1, x_2], dim=1)
        x = self.conv3(x)
        return x


# class C2fLayer(nn.Module):
#     """C2f layer with 3 convs, named C2f in YOLOv8"""
#
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  num_blocks=1,
#                  shortcut=False,
#                  expansion=0.5,
#                  depthwise=False,
#                  bias=False,
#                  act="silu"):
#         super(C2fLayer, self).__init__()
#         self.c = int(out_channels * expansion)  # hidden channels
#         self.conv1 = BaseConv(
#             in_channels, 2 * self.c, ksize=1, stride=1, bias=bias, act=act)
#         self.conv2 = BaseConv(
#             (2 + num_blocks) * self.c,
#             out_channels,
#             ksize=1,
#             stride=1,
#             bias=bias,
#             act=act)
#         self.bottlenecks = nn.Sequential(*[
#             BottleNeck(
#                 self.c,
#                 self.c,
#                 shortcut=shortcut,
#                 kernel_sizes=(3, 3),
#                 expansion=1.0,
#                 depthwise=depthwise,
#                 bias=bias,
#                 act=act) for _ in range(num_blocks)
#         ])
#
#     def forward(self, x):
#         y = list(self.conv1(x).split((self.c, self.c), 1))
#         y.extend(m(y[-1]) for m in self.bottlenecks)
#         return self.conv2(torch.concat(y, 1))
#

@register
@serializable
class CSPDarkNet(nn.Module):
    """
    CSPDarkNet backbone.
    Args:
        arch (str): Architecture of CSPDarkNet, from {P5, P6, X}, default as X,
            and 'X' means used in YOLOX, 'P5/P6' means used in YOLOv5.
        depth_mult (float): Depth multiplier, multiply number of channels in
            each layer, default as 1.0.
        width_mult (float): Width multiplier, multiply number of blocks in
            CSPLayer, default as 1.0.
        depthwise (bool): Whether to use depth-wise conv layer.
        act (str): Activation function type, default as 'silu'.
        return_idx (list): Index of stages whose feature maps are returned.
    """

    __shared__ = ['depth_mult', 'width_mult', 'act', 'trt']

    # in_channels, out_channels, num_blocks, add_shortcut, use_spp(use_sppf)
    # 'X' means setting used in YOLOX, 'P5/P6' means setting used in YOLOv5.
    arch_settings = {
        'X': [[64, 128, 3, True, False], [128, 256, 9, True, False],
              [256, 512, 9, True, False], [512, 1024, 3, False, True]],
        'P5': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 9, True, False], [512, 1024, 3, True, True]],
        'P6': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 9, True, False], [512, 768, 3, True, False],
               [768, 1024, 3, True, True]],
    }

    def __init__(self,
                 arch='X',
                 depth_mult=1.0,
                 width_mult=1.0,
                 depthwise=False,
                 act='silu',
                 trt=False,
                 return_idx=[2, 3, 4]):
        super(CSPDarkNet, self).__init__()
        self.arch = arch
        self.return_idx = return_idx
        Conv = DWConv if depthwise else BaseConv
        arch_setting = self.arch_settings[arch]
        base_channels = int(arch_setting[0][0] * width_mult)

        # Note: differences between the latest YOLOv5 and the original YOLOX
        # 1. self.stem, use SPPF(in YOLOv5) or SPP(in YOLOX)
        # 2. use SPPF(in YOLOv5) or SPP(in YOLOX)
        # 3. put SPPF before(YOLOv5) or SPP after(YOLOX) the last cspdark block's CSPLayer
        # 4. whether SPPF(SPP)'CSPLayer add shortcut, True in YOLOv5, False in YOLOX
        if arch in ['P5', 'P6']:
            # in the latest YOLOv5, use Conv stem, and SPPF (fast, only single spp kernal size)
            self.stem = Conv(
                3, base_channels, ksize=6, stride=2, bias=False, act=act)
            spp_kernal_sizes = 5
        elif arch in ['X']:
            # in the original YOLOX, use Focus stem, and SPP (three spp kernal sizes)
            self.stem = Focus(
                3, base_channels, ksize=3, stride=1, bias=False, act=act)
            spp_kernal_sizes = (5, 9, 13)
        else:
            raise AttributeError("Unsupported arch type: {}".format(arch))

        _out_channels = [base_channels]
        layers_num = 1
        self.csp_dark_blocks = []

        for i, (in_channels, out_channels, num_blocks, shortcut,
                use_spp) in enumerate(arch_setting):
            in_channels = int(in_channels * width_mult)
            out_channels = int(out_channels * width_mult)
            _out_channels.append(out_channels)
            num_blocks = max(round(num_blocks * depth_mult), 1)
            stage = []

            conv_layer = Conv(in_channels, out_channels, 3, 2, bias=False, act=act)
            self.add_module('layers{}_stage{}_conv_layer'.format(layers_num, i + 1), conv_layer)
            stage.append(conv_layer)
            layers_num += 1

            if use_spp and arch in ['X']:
                # in YOLOX use SPPLayer
                spp_layer = SPPLayer(out_channels,
                                     out_channels,
                                     kernel_sizes=spp_kernal_sizes,
                                     bias=False,
                                     act=act)
                self.add_module('layers{}_stage{}_spp_layer'.format(layers_num, i + 1), spp_layer)
                stage.append(spp_layer)
                layers_num += 1

            csp_layer = CSPLayer(out_channels,
                                 out_channels,
                                 num_blocks=num_blocks,
                                 shortcut=shortcut,
                                 depthwise=depthwise,
                                 bias=False,
                                 act=act)
            self.add_module('layers{}_stage{}_csp_layer'.format(layers_num, i + 1), csp_layer)
            stage.append(csp_layer)
            layers_num += 1

            if use_spp and arch in ['P5', 'P6']:
                # in latest YOLOv5 use SPPFLayer instead of SPPLayer
                sppf_layer = SPPFLayer(out_channels,
                                       out_channels,
                                       ksize=5,
                                       bias=False,
                                       act=act)
                self.add_module('layers{}_stage{}_sppf_layer'.format(layers_num, i + 1), sppf_layer)
                stage.append(sppf_layer)
                layers_num += 1

            self.csp_dark_blocks.append(nn.Sequential(*stage))

        self._out_channels = [_out_channels[i] for i in self.return_idx]
        self.strides = [[2, 4, 8, 16, 32, 64][i] for i in self.return_idx]

    def forward(self, inputs):
        x = inputs['image']
        outputs = []
        x = self.stem(x)
        for i, layer in enumerate(self.csp_dark_blocks):
            x = layer(x)
            if i + 1 in self.return_idx:
                outputs.append(x)
        return outputs

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=c, stride=s)
            for c, s in zip(self._out_channels, self.strides)
        ]


# @register
# @serializable
# class YOLOv8CSPDarkNet(nn.Module):
#     """
#     YOLOv8 CSPDarkNet backbone.
#     diff with YOLOv5 CSPDarkNet:
#     1. self.stem ksize 3 rather than 6 in YOLOv5
#     2. use C2fLayer rather than CSPLayer in YOLOv5
#     3. num_blocks [3,6,6,3] rather than [3,6,9,3] in YOLOv5
#     4. channels each stages
#
#     Args:
#         arch (str): Architecture of YOLOv8 CSPDarkNet, from {P5, P6}
#         depth_mult (float): Depth multiplier, multiply number of channels in
#             each layer, default as 1.0.
#         width_mult (float): Width multiplier, multiply number of blocks in
#             CSPLayer/C2fLayer, default as 1.0.
#         depthwise (bool): Whether to use depth-wise conv layer.
#         act (str): Activation function type, default as 'silu'.
#         return_idx (list): Index of stages whose feature maps are returned.
#     """
#
#     __shared__ = ['arch', 'depth_mult', 'width_mult', 'act', 'trt']
#
#     # in_channels, out_channels, num_blocks, add_shortcut, use_sppf
#     # Note: last stage's out channels are different
#     arch_settings = {
#         'n': [[64, 128, 3, True, False], [128, 256, 6, True, False],
#               [256, 512, 6, True, False], [512, 1024, 3, True, True]],
#         's': [[64, 128, 3, True, False], [128, 256, 6, True, False],
#               [256, 512, 6, True, False], [512, 1024, 3, True, True]],
#         'm': [[64, 128, 3, True, False], [128, 256, 6, True, False],
#               [256, 512, 6, True, False], [512, 768, 3, True, True]],  # 768
#         'l': [[64, 128, 3, True, False], [128, 256, 6, True, False],
#               [256, 512, 6, True, False], [512, 512, 3, True, True]],  # 512
#         'x': [[64, 128, 3, True, False], [128, 256, 6, True, False],
#               [256, 512, 6, True, False], [512, 512, 3, True, True]],  # 512
#     }
#
#     def __init__(self,
#                  arch='L',
#                  depth_mult=1.0,
#                  width_mult=1.0,
#                  depthwise=False,
#                  act='silu',
#                  trt=False,
#                  return_idx=[2, 3, 4]):
#         super(YOLOv8CSPDarkNet, self).__init__()
#         self.arch = arch.lower()
#         self.return_idx = return_idx
#         Conv = DWConv if depthwise else BaseConv
#
#         arch_setting = self.arch_settings[self.arch]
#         base_channels = int(arch_setting[0][0] * width_mult)
#
#         self.stem = Conv(
#             3, base_channels, ksize=3, stride=2, bias=False, act=act)
#
#         _out_channels = [base_channels]
#         layers_num = 1
#         self.csp_dark_blocks = []
#
#         for i, (in_channels, out_channels, num_blocks, shortcut,
#                 use_sppf) in enumerate(arch_setting):
#             in_channels = int(in_channels * width_mult)
#             out_channels = int(out_channels * width_mult)
#             _out_channels.append(out_channels)
#             num_blocks = max(round(num_blocks * depth_mult), 1)
#             stage = []
#
#             conv_layer = Conv(in_channels, out_channels, 3, 2, bias=False, act=act)
#             self.add_module('layers{}_stage{}_conv_layer'.format(layers_num, i + 1), conv_layer)
#             stage.append(conv_layer)
#             layers_num += 1
#
#             c2f_layer = C2fLayer(
#                 out_channels,
#                 out_channels,
#                 num_blocks=num_blocks,
#                 shortcut=shortcut,
#                 depthwise=depthwise,
#                 bias=False,
#                 act=act)
#             self.add_module('layers{}_stage{}_c2f_layer'.format(layers_num, i + 1), c2f_layer)
#             stage.append(c2f_layer)
#             layers_num += 1
#
#             if use_sppf:
#                 sppf_layer = SPPFLayer(
#                     out_channels,
#                     out_channels,
#                     ksize=5,
#                     bias=False,
#                     act=act)
#                 self.add_module('layers{}_stage{}_sppf_layer'.format(layers_num, i + 1), sppf_layer)
#                 stage.append(sppf_layer)
#                 layers_num += 1
#
#             self.csp_dark_blocks.append(nn.Sequential(*stage))
#
#         self._out_channels = [_out_channels[i] for i in self.return_idx]
#         self.strides = [[2, 4, 8, 16, 32, 64][i] for i in self.return_idx]
#
#     def forward(self, inputs):
#         x = inputs['image']
#         outputs = []
#         x = self.stem(x)
#         for i, layer in enumerate(self.csp_dark_blocks):
#             x = layer(x)
#             if i + 1 in self.return_idx:
#                 outputs.append(x)
#         return outputs
#
#     @property
#     def out_shape(self):
#         return [
#             ShapeSpec(
#                 channels=c, stride=s)
#             for c, s in zip(self._out_channels, self.strides)
#         ]
