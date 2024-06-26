# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from picodet.core.workspace import register
from ..bbox_utils import iou_similarity, batch_iou_similarity
from ..bbox_utils import bbox_center
from .utils import (check_points_inside_bboxes, compute_max_iou_anchor,
                    compute_max_iou_gt)

__all__ = ['ATSSAssigner']


@register
class ATSSAssigner(nn.Module):
    """Bridging the Gap Between Anchor-based and Anchor-free Detection
     via Adaptive Training Sample Selection
    """
    __shared__ = ['num_classes']

    def __init__(self,
                 topk=9,
                 num_classes=80,
                 force_gt_matching=False,
                 eps=1e-9):
        super(ATSSAssigner, self).__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.force_gt_matching = force_gt_matching
        self.eps = eps

    def _gather_topk_pyramid(self, gt2anchor_distances, num_anchors_list,
                             pad_gt_mask):
        gt2anchor_distances_list = torch.split(
            gt2anchor_distances, num_anchors_list, dim=-1)
        num_anchors_index = np.cumsum(num_anchors_list).tolist()
        num_anchors_index = [0, ] + num_anchors_index[:-1]
        is_in_topk_list = []
        topk_idxs_list = []
        for distances, anchors_index in zip(gt2anchor_distances_list,
                                            num_anchors_index):
            num_anchors = distances.shape[-1]
            _, topk_idxs = torch.topk(
                distances, self.topk, dim=-1, largest=False)
            topk_idxs_list.append(topk_idxs + anchors_index)
            is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(
                dim=-2).astype(gt2anchor_distances.dtype)
            is_in_topk_list.append(is_in_topk * pad_gt_mask)
        is_in_topk_list = torch.concat(is_in_topk_list, dim=-1)
        topk_idxs_list = torch.concat(topk_idxs_list, dim=-1)
        return is_in_topk_list, topk_idxs_list

    @torch.no_grad()
    def forward(self,
                anchor_bboxes,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                pad_gt_mask,
                bg_index,
                gt_scores=None,
                pred_bboxes=None):
        r"""This code is based on
            https://github.com/fcjian/TOOD/blob/master/mmdet/core/bbox/assigners/atss_assigner.py

        The assignment is done in following steps
        1. compute iou between all bbox (bbox of all pyramid levels) and gt
        2. compute center distance between all bbox and gt
        3. on each pyramid level, for each gt, select k bbox whose center
           are closest to the gt center, so we total select k*l bbox as
           candidates for each gt
        4. get corresponding iou for the these candidates, and compute the
           mean and std, set mean + std as the iou threshold
        5. select these candidates whose iou are greater than or equal to
           the threshold as positive
        6. limit the positive sample's center in gt
        7. if an anchor box is assigned to multiple gts, the one with the
           highest iou will be selected.
        Args:
            anchor_bboxes (Tensor, float32): pre-defined anchors, shape(L, 4),
                    "xmin, xmax, ymin, ymax" format
            num_anchors_list (List): num of anchors in each level
            gt_labels (Tensor, int64|int32): Label of gt_bboxes, shape(B, n, 1)
            gt_bboxes (Tensor, float32): Ground truth bboxes, shape(B, n, 4)
            pad_gt_mask (Tensor, float32): 1 means bbox, 0 means no bbox, shape(B, n, 1)
            bg_index (int): background index
            gt_scores (Tensor|None, float32) Score of gt_bboxes,
                    shape(B, n, 1), if None, then it will initialize with one_hot label
            pred_bboxes (Tensor, float32, optional): predicted bounding boxes, shape(B, L, 4)
        Returns:
            assigned_labels (Tensor): (B, L)
            assigned_bboxes (Tensor): (B, L, 4)
            assigned_scores (Tensor): (B, L, C), if pred_bboxes is not None, then output ious
        """
        assert gt_labels.ndim == gt_bboxes.ndim and \
               gt_bboxes.ndim == 3

        num_anchors, _ = anchor_bboxes.shape
        batch_size, num_max_boxes, _ = gt_bboxes.shape

        # negative batch
        if num_max_boxes == 0:
            assigned_labels = torch.full(
                [batch_size, num_anchors], bg_index, dtype='int32')
            assigned_bboxes = torch.zeros([batch_size, num_anchors, 4])
            assigned_scores = torch.zeros(
                [batch_size, num_anchors, self.num_classes])
            return assigned_labels, assigned_bboxes, assigned_scores

        # 1. compute iou between gt and anchor bbox, [B, n, L]
        ious = iou_similarity(gt_bboxes.reshape([-1, 4]), anchor_bboxes)
        ious = ious.reshape([batch_size, -1, num_anchors])

        # 2. compute center distance between all anchors and gt, [B, n, L]
        gt_centers = bbox_center(gt_bboxes.reshape([-1, 4])).unsqueeze(1)
        anchor_centers = bbox_center(anchor_bboxes)
        gt2anchor_distances = (gt_centers - anchor_centers.unsqueeze(0)) \
            .norm(2, dim=-1).reshape([batch_size, -1, num_anchors])

        # 3. on each pyramid level, selecting topk closest candidates
        # based on the center distance, [B, n, L]
        is_in_topk, topk_idxs = self._gather_topk_pyramid(
            gt2anchor_distances, num_anchors_list, pad_gt_mask)

        # 4. get corresponding iou for the these candidates, and compute the
        # mean and std, 5. set mean + std as the iou threshold
        iou_candidates = ious * is_in_topk
        iou_threshold = torch.index_sample(
            iou_candidates.flatten(stop_dim=-2),
            topk_idxs.flatten(stop_dim=-2))
        iou_threshold = iou_threshold.reshape([batch_size, num_max_boxes, -1])
        iou_threshold = iou_threshold.mean(dim=-1, keepdim=True) + \
                        iou_threshold.std(dim=-1, keepdim=True)
        is_in_topk = torch.where(iou_candidates > iou_threshold, is_in_topk,
                                 torch.zeros_like(is_in_topk))

        # 6. check the positive sample's center in gt, [B, n, L]
        is_in_gts = check_points_inside_bboxes(anchor_centers, gt_bboxes)

        # select positive sample, [B, n, L]
        mask_positive = is_in_topk * is_in_gts * pad_gt_mask

        # 7. if an anchor box is assigned to multiple gts,
        # the one with the highest iou will be selected.
        mask_positive_sum = mask_positive.sum(dim=-2)
        if mask_positive_sum.max() > 1:
            mask_multiple_gts = (mask_positive_sum.unsqueeze(1) > 1).tile(
                [1, num_max_boxes, 1])
            is_max_iou = compute_max_iou_anchor(ious)
            mask_positive = torch.where(mask_multiple_gts, is_max_iou,
                                        mask_positive)
            mask_positive_sum = mask_positive.sum(dim=-2)
        # 8. make sure every gt_bbox matches the anchor
        if self.force_gt_matching:
            is_max_iou = compute_max_iou_gt(ious) * pad_gt_mask
            mask_max_iou = (is_max_iou.sum(-2, keepdim=True) == 1).tile(
                [1, num_max_boxes, 1])
            mask_positive = torch.where(mask_max_iou, is_max_iou,
                                        mask_positive)
            mask_positive_sum = mask_positive.sum(dim=-2)
        assigned_gt_index = mask_positive.argmax(dim=-2)

        # assigned target
        batch_ind = torch.arange(
            end=batch_size, dtype=gt_labels.dtype).unsqueeze(-1)
        assigned_gt_index = assigned_gt_index + batch_ind * num_max_boxes
        assigned_labels = torch.gather(
            gt_labels.flatten(), assigned_gt_index.flatten(), dim=0)
        assigned_labels = assigned_labels.reshape([batch_size, num_anchors])
        assigned_labels = torch.where(
            mask_positive_sum > 0, assigned_labels,
            torch.full_like(assigned_labels, bg_index))

        assigned_bboxes = torch.gather(
            gt_bboxes.reshape([-1, 4]), assigned_gt_index.flatten(), dim=0)
        assigned_bboxes = assigned_bboxes.reshape([batch_size, num_anchors, 4])

        assigned_scores = F.one_hot(assigned_labels, self.num_classes + 1)
        ind = list(range(self.num_classes + 1))
        ind.remove(bg_index)
        assigned_scores = torch.index_select(
            assigned_scores, torch.to_tensor(ind), dim=-1)
        if pred_bboxes is not None:
            # assigned iou
            ious = batch_iou_similarity(gt_bboxes, pred_bboxes) * mask_positive
            ious = ious.max(dim=-2).unsqueeze(-1)
            assigned_scores *= ious
        elif gt_scores is not None:
            gather_scores = torch.gather(
                gt_scores.flatten(), assigned_gt_index.flatten(), dim=0)
            gather_scores = gather_scores.reshape([batch_size, num_anchors])
            gather_scores = torch.where(mask_positive_sum > 0, gather_scores,
                                        torch.zeros_like(gather_scores))
            assigned_scores *= gather_scores.unsqueeze(-1)

        return assigned_labels, assigned_bboxes, assigned_scores
