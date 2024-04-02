#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

# The code is based on 
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/anchor_generator.py

import math

import torch
import torch.nn as nn
import numpy as np
from picodet.core.workspace import register

__all__ = ['AnchorGenerator', 'RetinaAnchorGenerator', 'S2ANetAnchorGenerator']




@register
class AnchorGenerator(nn.Module):
    """
    Generate anchors according to the feature maps

    Args:
        anchor_sizes (list[float] | list[list[float]]): The anchor sizes at 
            each feature point. list[float] means all feature levels share the 
            same sizes. list[list[float]] means the anchor sizes for 
            each level. The sizes stand for the scale of input size.
        aspect_ratios (list[float] | list[list[float]]): The aspect ratios at
            each feature point. list[float] means all feature levels share the
            same ratios. list[list[float]] means the aspect ratios for
            each level.
        strides (list[float]): The strides of feature maps which generate 
            anchors
        offset (float): The offset of the coordinate of anchors, default 0.
        
    """

    def __init__(self,
                 anchor_sizes=[32, 64, 128, 256, 512],
                 aspect_ratios=[0.5, 1.0, 2.0],
                 strides=[16.0],
                 variance=[1.0, 1.0, 1.0, 1.0],
                 offset=0.):
        super(AnchorGenerator, self).__init__()
        self.anchor_sizes = anchor_sizes
        self.aspect_ratios = aspect_ratios
        self.strides = strides
        self.variance = variance
        self.cell_anchors = self._calculate_anchors(len(strides))
        self.offset = offset

    def _broadcast_params(self, params, num_features):
        if not isinstance(params[0], (list, tuple)):  # list[float]
            return [params] * num_features
        if len(params) == 1:
            return list(params) * num_features
        return params

    def generate_cell_anchors(self, sizes, aspect_ratios):
        anchors = []
        for size in sizes:
            area = size**2.0
            for aspect_ratio in aspect_ratios:
                w = math.sqrt(area / aspect_ratio)
                h = aspect_ratio * w
                x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
                anchors.append([x0, y0, x1, y1])
        return torch.tensor(anchors)

    def _calculate_anchors(self, num_features):
        sizes = self._broadcast_params(self.anchor_sizes, num_features)
        aspect_ratios = self._broadcast_params(self.aspect_ratios, num_features)
        cell_anchors = [
            self.generate_cell_anchors(s, a)
            for s, a in zip(sizes, aspect_ratios)
        ]
        [
            self.register_buffer(
                str(index), t, persistent=False) for index,t in enumerate(cell_anchors)
        ]
        return cell_anchors

    def _create_grid_offsets(self, size, stride, offset):
        grid_height, grid_width = size[0], size[1]
        shifts_x = torch.arange(
            offset * stride, grid_width * stride, step=stride, dtype=torch.float32)
        shifts_y = torch.arange(
            offset * stride, grid_height * stride, step=stride, dtype=torch.float32)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = torch.reshape(shift_x, [-1])
        shift_y = torch.reshape(shift_y, [-1])
        return shift_x, shift_y

    def _grid_anchors(self, grid_sizes):
        anchors = []
        for size, stride, base_anchors in zip(grid_sizes, self.strides,
                                              self.cell_anchors):
            shift_x, shift_y = self._create_grid_offsets(size, stride,
                                                         self.offset)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
            shifts = torch.reshape(shifts, [-1, 1, 4])
            base_anchors = torch.reshape(base_anchors, [1, -1, 4])

            anchors.append(torch.reshape(shifts + base_anchors, [-1, 4]))

        return anchors

    def forward(self, input):
        grid_sizes = [feature_map.shape[-2:] for feature_map in input]
        anchors_over_all_feature_maps = self._grid_anchors(grid_sizes)
        return anchors_over_all_feature_maps

    @property
    def num_anchors(self):
        """
        Returns:
            int: number of anchors at every pixel
                location, on that feature map.
                For example, if at every pixel we use anchors of 3 aspect
                ratios and 5 sizes, the number of anchors is 15.
                For FPN models, `num_anchors` on every feature map is the same.
        """
        return len(self.cell_anchors[0])


@register
class RetinaAnchorGenerator(AnchorGenerator):
    def __init__(self,
                 octave_base_scale=4,
                 scales_per_octave=3,
                 aspect_ratios=[0.5, 1.0, 2.0],
                 strides=[8.0, 16.0, 32.0, 64.0, 128.0],
                 variance=[1.0, 1.0, 1.0, 1.0],
                 offset=0.0):
        anchor_sizes = []
        for s in strides:
            anchor_sizes.append([
                s * octave_base_scale * 2**(i/scales_per_octave) \
                for i in range(scales_per_octave)])
        super(RetinaAnchorGenerator, self).__init__(
            anchor_sizes=anchor_sizes,
            aspect_ratios=aspect_ratios,
            strides=strides,
            variance=variance,
            offset=offset)


@register
class S2ANetAnchorGenerator(nn.Module):
    """
    AnchorGenerator by paddle
    """

    def __init__(self, base_size, scales, ratios, scale_major=True, ctr=None):
        super(S2ANetAnchorGenerator, self).__init__()
        self.base_size = base_size
        self.scales = torch.tensor(scales)
        self.ratios = torch.tensor(ratios)
        self.scale_major = scale_major
        self.ctr = ctr
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_base_anchors(self):
        return self.base_anchors.shape[0]

    def gen_base_anchors(self):
        w = self.base_size
        h = self.base_size
        if self.ctr is None:
            x_ctr = 0.5 * (w - 1)
            y_ctr = 0.5 * (h - 1)
        else:
            x_ctr, y_ctr = self.ctr

        h_ratios = torch.sqrt(self.ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = (w * w_ratios[:] * self.scales[:]).reshape([-1])
            hs = (h * h_ratios[:] * self.scales[:]).reshape([-1])
        else:
            ws = (w * self.scales[:] * w_ratios[:]).reshape([-1])
            hs = (h * self.scales[:] * h_ratios[:]).reshape([-1])

        base_anchors = torch.stack(
            [
                x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
                x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)
            ],
            dim=-1)
        base_anchors = torch.round(base_anchors)
        return base_anchors

    def _meshgrid(self, x, y, row_major=True):
        yy, xx = torch.meshgrid(y, x)
        yy = yy.reshape([-1])
        xx = xx.reshape([-1])
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def forward(self, featmap_size, stride=16):
        # featmap_size*stride project it to original area

        feat_h = featmap_size[0]
        feat_w = featmap_size[1]
        shift_x = torch.arange(0, feat_w, 1, dtype=torch.int32) * stride
        shift_y = torch.arange(0, feat_h, 1, dtype=torch.int32) * stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)

        all_anchors = self.base_anchors[:, :] + shifts[:, :]
        all_anchors = all_anchors.cast(torch.float32).reshape(
            [feat_h * feat_w, 4])
        all_anchors = self.rect2rbox(all_anchors)
        return all_anchors

    def valid_flags(self, featmap_size, valid_size):
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros([feat_w], dtype=torch.int32)
        valid_y = torch.zeros([feat_h], dtype=torch.int32)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        valid = torch.reshape(valid, [-1, 1])
        valid = torch.expand(valid, [-1, self.num_base_anchors]).reshape([-1])
        return valid

    def rect2rbox(self, bboxes):
        """
        :param bboxes: shape (L, 4) (xmin, ymin, xmax, ymax)
        :return: dbboxes: shape (L, 5) (x_ctr, y_ctr, w, h, angle)
        """
        x1, y1, x2, y2 = torch.split(bboxes, 4, dim=-1)

        x_ctr = (x1 + x2) / 2.0
        y_ctr = (y1 + y2) / 2.0
        edges1 = torch.abs(x2 - x1)
        edges2 = torch.abs(y2 - y1)

        rbox_w = torch.maximum(edges1, edges2)
        rbox_h = torch.minimum(edges1, edges2)

        # set angle
        inds = edges1 < edges2
        inds = inds.to(torch.float32)
        rboxes_angle = inds * np.pi / 2.0

        rboxes = torch.concat(
            (x_ctr, y_ctr, rbox_w, rbox_h, rboxes_angle), dim=-1)
        return rboxes
