# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from picodet.core.workspace import register, serializable
from ..layers import MultiClassNMS

from ..bbox_utils import batch_distance2bbox
from ..bbox_utils import bbox_iou
from ..assigners.utils import generate_anchors_for_grid_cell
from ..backbones.csp_darknet import BaseConv
from ..ops import get_static_shape

__all__ = ['YOLOv8Head']


@register
class YOLOv8Head(nn.Module):
    __shared__ = ['num_classes', 'eval_size', 'trt', 'exclude_nms']
    __inject__ = ['assigner', 'nms']

    def __init__(self,
                 in_channels=[256, 512, 1024],
                 num_classes=80,
                 act='silu',
                 fpn_strides=[8, 16, 32],
                 grid_cell_scale=5.0,
                 grid_cell_offset=0.5,
                 reg_max=16,
                 use_varifocal_loss=False,
                 assigner='TaskAlignedAssigner',
                 nms='MultiClassNMS',
                 eval_size=None,
                 loss_weight={
                     'class': 1.0,
                     'iou': 2.5,
                     'dfl': 0.5,
                 },
                 trt=False,
                 exclude_nms=False):
        super(YOLOv8Head, self).__init__()
        assert len(in_channels) > 0, "len(in_channels) should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.reg_max = reg_max
        self.use_varifocal_loss = use_varifocal_loss
        self.assigner = assigner
        self.nms = nms
        if isinstance(self.nms, MultiClassNMS) and trt:
            self.nms.trt = trt
        self.eval_size = eval_size
        self.loss_weight = loss_weight
        self.exclude_nms = exclude_nms

        # cls loss
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([1.0]), reduction="mean")

        # pred head
        c2 = max((16, in_channels[0] // 4, self.reg_max * 4))
        c3 = max(in_channels[0], self.num_classes)
        self.conv_reg = nn.ModuleList()
        self.conv_cls = nn.ModuleList()
        for in_c in self.in_channels:
            self.conv_reg.append(
                nn.Sequential(*[
                    BaseConv(
                        in_c, c2, 3, 1, act=act),
                    BaseConv(
                        c2, c2, 3, 1, act=act),
                    nn.Conv2d(
                        c2,
                        self.reg_max * 4,
                        1,
                        bias=True),
                ]))
            self.conv_cls.append(
                nn.Sequential(*[
                    BaseConv(
                        in_c, c3, 3, 1, act=act),
                    BaseConv(
                        c3, c3, 3, 1, act=act),
                    nn.Conv2d(
                        c3,
                        self.num_classes,
                        1,
                        bias=True),
                ]))
        # projection conv
        self.dfl_conv = nn.Conv2d(self.reg_max, 1, 1, bias=False)
        self.dfl_conv.skip_quant = True
        self.proj = torch.linspace(0, self.reg_max - 1, self.reg_max)
        self.dfl_conv.weight = nn.Parameter(self.proj.reshape([1, self.reg_max, 1, 1]))
        self.dfl_conv.weight.stop_gradient = True

        # self._init_bias()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def _init_bias(self):
        for a, b, s in zip(self.conv_reg, self.conv_cls, self.fpn_strides):
            a[-1].bias.set_value(1.0)  # box
            b[-1].bias[:self.num_classes] = math.log(5 / self.num_classes /
                                                     (640 / s) ** 2)

    def forward(self, feats, targets=None):
        if self.training:
            return self.forward_train(feats, targets)
        else:
            return self.forward_eval(feats)

    def forward_train(self, feats, targets):
        anchors, anchor_points, num_anchors_list, stride_tensor = \
            generate_anchors_for_grid_cell(
                feats, self.fpn_strides, self.grid_cell_scale,
                self.grid_cell_offset)

        cls_list, reg_list = [], []
        for i, feat in enumerate(feats):
            pred_reg = self.conv_reg[i](feat)
            pred_cls = self.conv_cls[i](feat)
            reg_list.append(pred_reg.flatten(2).permute(0, 2, 1))
            cls_list.append(pred_cls.flatten(2).permute(0, 2, 1))
        cls_concat = torch.concat(cls_list, dim=1)
        reg_concat = torch.concat(reg_list, dim=1)
        return self.get_loss([
            cls_concat, reg_concat, anchors, anchor_points, num_anchors_list,
            stride_tensor
        ], targets)

    def forward_eval(self, feats):
        anchor_points, stride_tensor = self._generate_anchors(feats)

        cls_list, reg_list = [], []
        for i, feat in enumerate(feats):
            b, _, h, w = feat.shape
            l = h * w
            pred_reg = self.conv_reg[i](feat)
            pred_cls = self.conv_cls[i](feat)

            pred_reg = pred_reg.reshape([-1, 4, self.reg_max, l]).permute(0, 2, 1, 3)
            pred_reg = self.dfl_conv(F.softmax(pred_reg, dim=1))
            cls_score = F.sigmoid(pred_cls)

            cls_list.append(cls_score.reshape([b, self.num_classes, l]))
            reg_list.append(pred_reg.reshape([b, 4, l]))
        cls_concat = torch.concat(cls_list, dim=-1)
        reg_concat = torch.concat(reg_list, dim=-1)
        return cls_concat, reg_concat, anchor_points, stride_tensor

    def _generate_anchors(self, feats=None, dtype=torch.float32):
        # just use in eval time
        anchor_points = []
        stride_tensor = []
        for i, stride in enumerate(self.fpn_strides):
            if feats is not None:
                _, _, h, w = feats[i].shape
            else:
                h = int(self.eval_size[0] / stride)
                w = int(self.eval_size[1] / stride)
            shift_x = torch.arange(end=w) + self.grid_cell_offset
            shift_y = torch.arange(end=h) + self.grid_cell_offset
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
            anchor_point = torch.stack([shift_x, shift_y], dim=-1)
            anchor_points.append(anchor_point.reshape([-1, 2]))
            stride_tensor.append(torch.full([h * w, 1], stride, dtype=dtype))
        anchor_points = torch.concat(anchor_points)
        stride_tensor = torch.concat(stride_tensor)
        return anchor_points, stride_tensor

    @staticmethod
    def _focal_loss(score, label, alpha=0.25, gamma=2.0):
        weight = (score - label).pow(gamma)
        if alpha > 0:
            alpha_t = alpha * label + (1 - alpha) * (1 - label)
            weight *= alpha_t
        loss = F.binary_cross_entropy(
            score, label, weight=weight, reduction='sum')
        return loss

    @staticmethod
    def _varifocal_loss(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
        loss = F.binary_cross_entropy(
            pred_score, gt_score, weight=weight, reduction='sum')
        return loss

    def _bbox_decode(self, anchor_points, pred_dist):
        b, l, _ = get_static_shape(pred_dist)
        pred_dist = F.softmax(pred_dist.reshape([b, l, 4, self.reg_max
                                                 ])).matmul(self.proj)
        return batch_distance2bbox(anchor_points, pred_dist)

    def _bbox2distance(self, points, bbox, reg_max):
        x1y1, x2y2 = torch.split(bbox, 2, -1)
        lt = points - x1y1
        rb = x2y2 - points
        return torch.concat([lt, rb], -1).clip(0, reg_max - 0.01)

    def _df_loss(self, pred_dist, target):
        target_left = target.long()
        target_right = target_left + 1
        weight_left = target_right - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(
            pred_dist, target_left, reduction='none') * weight_left
        loss_right = F.cross_entropy(
            pred_dist, target_right, reduction='none') * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)

    def _bbox_loss(self, pred_dist, pred_bboxes, anchor_points, assigned_labels,
                   assigned_bboxes, assigned_scores, assigned_scores_sum):
        # select positive samples mask
        mask_positive = (assigned_labels != self.num_classes)
        num_pos = mask_positive.sum()
        # pos/neg loss
        if num_pos > 0:
            # l1 + iou
            bbox_mask = mask_positive.unsqueeze(-1).tile([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(pred_bboxes,
                                                   bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = torch.masked_select(
                assigned_bboxes, bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(
                assigned_scores.sum(-1), mask_positive).unsqueeze(-1)

            # loss_l1 just see if train well
            loss_l1 = F.l1_loss(pred_bboxes_pos, assigned_bboxes_pos)

            # ciou loss
            iou = bbox_iou(
                pred_bboxes_pos, assigned_bboxes_pos, x1y1x2y2=False, ciou=True)
            loss_iou = ((1.0 - iou) * bbox_weight).sum() / assigned_scores_sum

            dist_mask = mask_positive.unsqueeze(-1).tile(
                [1, 1, (self.reg_max) * 4])
            pred_dist_pos = torch.masked_select(
                pred_dist, dist_mask).reshape([-1, 4, self.reg_max])
            assigned_ltrb = self._bbox2distance(anchor_points, assigned_bboxes,
                                                self.reg_max - 1)
            assigned_ltrb_pos = torch.masked_select(
                assigned_ltrb, bbox_mask).reshape([-1, 4])
            loss_dfl = self._df_loss(pred_dist_pos,
                                     assigned_ltrb_pos) * bbox_weight
            loss_dfl = loss_dfl.sum() / assigned_scores_sum
        else:
            loss_l1 = torch.zeros([1])
            loss_iou = torch.zeros([1])
            loss_dfl = pred_dist.sum() * 0.
        return loss_l1, loss_iou, loss_dfl

    def get_loss(self, head_outs, gt_meta):
        pred_scores, pred_distri, anchors, \
            anchor_points, num_anchors_list, stride_tensor = head_outs

        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self._bbox_decode(anchor_points_s, pred_distri)

        gt_labels = gt_meta['gt_class']
        gt_bboxes = gt_meta['gt_bbox']
        pad_gt_mask = gt_meta['pad_gt_mask']
        assigned_labels, assigned_bboxes, assigned_scores = \
            self.assigner(
                pred_scores.detach(),
                pred_bboxes.detach() * stride_tensor,
                anchor_points,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                pad_gt_mask,
                bg_index=self.num_classes)
        # rescale bbox
        assigned_bboxes /= stride_tensor
        # cls loss
        if self.use_varifocal_loss:
            one_hot_label = F.one_hot(assigned_labels,
                                      self.num_classes + 1)[..., :-1]
            loss_cls = self._varifocal_loss(pred_scores, assigned_scores,
                                            one_hot_label)
        else:
            loss_cls = self.bce(pred_scores, assigned_scores)

        assigned_scores_sum = assigned_scores.sum()
        # if torch.distributed.get_world_size() > 1:
        #     torch.distributed.all_reduce(assigned_scores_sum)
        #     assigned_scores_sum = torch.clip(
        #         assigned_scores_sum / torch.distributed.get_world_size(),
        #         min=1)
        # loss_cls /= assigned_scores_sum

        loss_l1, loss_iou, loss_dfl = \
            self._bbox_loss(pred_distri, pred_bboxes, anchor_points_s,
                            assigned_labels, assigned_bboxes, assigned_scores,
                            assigned_scores_sum)
        loss = self.loss_weight['class'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou + \
               self.loss_weight['dfl'] * loss_dfl
        out_dict = {
            'loss': loss,
            'loss_cls': loss_cls,
            'loss_iou': loss_iou,
            'loss_dfl': loss_dfl,
            'loss_l1': loss_l1,
        }
        return out_dict

    def post_process(self, head_outs, im_shape, scale_factor):
        pred_scores, reg_concat, anchor_points, stride_tensor = head_outs
        run_device = pred_scores.device
        anchor_points = anchor_points.to(run_device)
        stride_tensor = stride_tensor.to(run_device)

        pred_bboxes = batch_distance2bbox(anchor_points,
                                          reg_concat.permute(0, 2, 1))
        pred_bboxes *= stride_tensor
        # scale bbox to origin
        scale_y, scale_x = torch.split(scale_factor, [1,1], dim=-1)
        scale_factor = torch.concat(
            [scale_x, scale_y, scale_x, scale_y], dim=-1).reshape([-1, 1, 4])
        pred_bboxes /= scale_factor

        if self.exclude_nms:
            # `exclude_nms=True` just use in benchmark
            return pred_bboxes.sum(), pred_scores.sum()
        else:
            bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
            return bbox_pred, bbox_num
