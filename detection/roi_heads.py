"""
roi_heads.py  –  Modified for Oriented Bounding Box (OBB) prediction.

Changes vs original:
  • RoIHeads now optionally predicts an angle alongside class scores and box
    regression (controlled by angle_mode: None | 'regression' | 'multibin').
  • fastrcnn_loss extended to return an angle_loss when applicable.
  • postprocess_detections returns angles in the result dict when enabled.

angle_mode options
------------------
  None          – standard axis-aligned behaviour (drop-in replacement)
  'regression'  – one scalar angle per detection, smooth-L1 loss
  'multibin'    – num_bins class logits per detection, cross-entropy loss;
                  the predicted angle is bin_centre + optional residual.

The angle loss is *added to* the existing classification + box losses with a
tunable weight (angle_loss_weight).
"""

from typing import Dict, List, Optional, Tuple
import math

import torch
import torch.nn.functional as F
import torchvision
from torch import nn, Tensor
from torchvision.ops import boxes as box_ops, roi_align

from . import _utils as det_utils


# ═══════════════════════════════════════════════════════════════════════════════
# Loss function
# ═══════════════════════════════════════════════════════════════════════════════

def fastrcnn_loss(class_logits, box_regression, labels, regression_targets,
                  angle_pred=None, angle_targets=None,
                  angle_mode=None, num_bins=6):
    """
    Computes the loss for Faster R-CNN (extended for OBB).

    Returns
    -------
    classification_loss, box_loss[, angle_loss]
    angle_loss is 0 when angle_mode is None.
    """
    labels            = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(class_logits, labels)

    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction="sum",
    ) / labels.numel()

    # ── angle loss ────────────────────────────────────────────────────────────
    angle_loss = torch.tensor(0.0, device=class_logits.device)

    if angle_mode is not None and angle_pred is not None and angle_targets is not None:
        angle_targets_cat = torch.cat(angle_targets, dim=0)          # [N_total]

        if angle_mode == 'regression':
            # angle_pred: [N_total, 1]  – direct scalar regression
            angle_pred_pos = angle_pred[sampled_pos_inds_subset].squeeze(1)
            angle_tgt_pos  = angle_targets_cat[sampled_pos_inds_subset]
            angle_loss = F.smooth_l1_loss(angle_pred_pos, angle_tgt_pos,
                                          beta=1.0, reduction='sum') / (labels.numel() + 1e-6)

        elif angle_mode == 'multibin':
            # angle_pred: [N_total, num_bins]  – bin classification
            bin_size = 180.0 / num_bins   # degrees per bin
            # Convert continuous angle → bin index
            angle_tgt_pos  = angle_targets_cat[sampled_pos_inds_subset]
            # Map angles from [-90, 90) to [0, 180)
            angle_shifted   = (angle_tgt_pos % 180.0)
            bin_idx         = (angle_shifted / bin_size).long().clamp(0, num_bins - 1)
            angle_pred_pos  = angle_pred[sampled_pos_inds_subset]    # [P, num_bins]
            angle_loss = F.cross_entropy(angle_pred_pos, bin_idx)

    return classification_loss, box_loss, angle_loss


# ═══════════════════════════════════════════════════════════════════════════════
# RoIHeads
# ═══════════════════════════════════════════════════════════════════════════════

class RoIHeads(nn.Module):
    __annotations__ = {
        "box_coder":       det_utils.BoxCoder,
        "proposal_matcher": det_utils.Matcher,
        "fg_bg_sampler":   det_utils.BalancedPositiveNegativeSampler,
    }

    def __init__(
        self,
        box_roi_pool,
        box_head,
        box_predictor,
        # training
        fg_iou_thresh,
        bg_iou_thresh,
        batch_size_per_image,
        positive_fraction,
        bbox_reg_weights,
        # inference
        score_thresh,
        nms_thresh,
        detections_per_img,
        # OBB extras (all optional for drop-in compatibility)
        angle_predictor=None,
        angle_mode=None,       # None | 'regression' | 'multibin'
        num_bins=6,
        angle_loss_weight=1.0,
    ):
        super().__init__()

        self.box_similarity  = box_ops.box_iou
        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=False
        )
        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image, positive_fraction
        )
        if bbox_reg_weights is None:
            bbox_reg_weights = (10.0, 10.0, 5.0, 5.0)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_roi_pool  = box_roi_pool
        self.box_head      = box_head
        self.box_predictor = box_predictor

        self.score_thresh       = score_thresh
        self.nms_thresh         = nms_thresh
        self.detections_per_img = detections_per_img

        # OBB
        self.angle_predictor  = angle_predictor
        self.angle_mode       = angle_mode
        self.num_bins         = num_bins
        self.angle_loss_weight = angle_loss_weight

    # ── target assignment (unchanged) ─────────────────────────────────────────

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        matched_idxs = []
        labels = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(
                proposals, gt_boxes, gt_labels):

            if gt_boxes_in_image.numel() == 0:
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device)
                labels_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device)
            else:
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)
                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)
                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_image[bg_inds] = 0
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_image[ignore_inds] = -1

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def subsample(self, labels):
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
                zip(sampled_pos_inds, sampled_neg_inds)):
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0]
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def add_gt_proposals(self, proposals, gt_boxes):
        proposals = [torch.cat((proposal, gt_box))
                     for proposal, gt_box in zip(proposals, gt_boxes)]
        return proposals

    def check_targets(self, targets):
        if targets is None:
            raise ValueError("targets should not be None")
        if not all(["boxes" in t for t in targets]):
            raise ValueError("Every element of targets should have a boxes key")
        if not all(["labels" in t for t in targets]):
            raise ValueError("Every element of targets should have a labels key")

    def select_training_samples(self, proposals, targets):
        self.check_targets(targets)
        if targets is None:
            raise ValueError("targets should not be None")
        dtype  = proposals[0].dtype
        device = proposals[0].device

        gt_boxes   = [t["boxes"].to(dtype) for t in targets]
        gt_labels  = [t["labels"] for t in targets]
        gt_angles  = [t["angles"] for t in targets] if (
            self.angle_mode is not None and
            targets[0].get("angles") is not None
        ) else None

        proposals = self.add_gt_proposals(proposals, gt_boxes)

        matched_idxs, labels = self.assign_targets_to_proposals(
            proposals, gt_boxes, gt_labels)
        sampled_inds    = self.subsample(labels)
        matched_gt_boxes = []
        matched_gt_angles = [] if gt_angles is not None else None
        num_images = len(proposals)

        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id]   = proposals[img_id][img_sampled_inds]
            labels[img_id]      = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

            if gt_angles is not None:
                gt_ang = gt_angles[img_id]
                if gt_ang.numel() == 0:
                    gt_ang = torch.zeros((1,), dtype=dtype, device=device)
                matched_gt_angles.append(gt_ang[matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets, matched_gt_angles

    # ── post-processing (adds angle to result) ────────────────────────────────

    def postprocess_detections(self, class_logits, box_regression, proposals,
                                image_shapes, angle_pred=None):
        device      = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [b.shape[0] for b in proposals]
        pred_boxes  = self.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list  = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)
        if angle_pred is not None:
            pred_angles_list = angle_pred.split(boxes_per_image, 0)
        else:
            pred_angles_list = [None] * len(pred_boxes_list)

        all_boxes, all_scores, all_labels, all_angles = [], [], [], []

        for boxes, scores, image_shape, ang in zip(
                pred_boxes_list, pred_scores_list, image_shapes, pred_angles_list):

            boxes  = box_ops.clip_boxes_to_image(boxes, image_shape)
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            boxes  = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            boxes  = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # ── angles ────────────────────────────────────────────────────────
            if ang is not None and self.angle_mode == 'regression':
                # ang: [P, 1]  → replicate for each non-bg class
                ang = ang.reshape(-1, 1).expand(-1, num_classes - 1).reshape(-1)
            elif ang is not None and self.angle_mode == 'multibin':
                # Convert predicted bin → centre angle (degrees)
                bin_size = 180.0 / self.num_bins
                pred_bin = ang.argmax(dim=-1)            # [P]
                ang_deg  = pred_bin.float() * bin_size + bin_size / 2.0 - 90.0
                ang = ang_deg.unsqueeze(1).expand(-1, num_classes - 1).reshape(-1)
            else:
                ang = None

            inds   = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]
            if ang is not None:
                ang = ang[inds]

            keep   = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            if ang is not None:
                ang = ang[keep]

            keep   = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            keep   = keep[:self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            if ang is not None:
                ang = ang[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            all_angles.append(ang)

        return all_boxes, all_scores, all_labels, all_angles

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, features, proposals, image_shapes, targets=None):
        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                if t["boxes"].dtype not in floating_point_types:
                    raise TypeError(
                        f"target boxes must of float type, got {t['boxes'].dtype}")
                if t["labels"].dtype != torch.int64:
                    raise TypeError(
                        f"target labels must of int64 type, got {t['labels'].dtype}")

        if self.training:
            (proposals, matched_idxs,
             labels, regression_targets,
             angle_targets) = self.select_training_samples(proposals, targets)
        else:
            labels = regression_targets = matched_idxs = angle_targets = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        # ── angle prediction ──────────────────────────────────────────────────
        if self.angle_predictor is not None:
            angle_pred = self.angle_predictor(box_features)   # [N, 1] or [N, num_bins]
        else:
            angle_pred = None

        result: List[Dict[str, torch.Tensor]] = []
        losses = {}

        if self.training:
            if labels is None:
                raise ValueError("labels cannot be None")
            if regression_targets is None:
                raise ValueError("regression_targets cannot be None")

            loss_classifier, loss_box_reg, angle_loss = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets,
                angle_pred=angle_pred,
                angle_targets=angle_targets,
                angle_mode=self.angle_mode,
                num_bins=self.num_bins,
            )
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg":    loss_box_reg,
                "loss_angle":      self.angle_loss_weight * angle_loss,
            }
        else:
            boxes, scores, det_labels, angles = self.postprocess_detections(
                class_logits, box_regression, proposals, image_shapes,
                angle_pred=angle_pred,
            )
            num_images = len(boxes)
            for i in range(num_images):
                entry = {
                    "boxes":  boxes[i],
                    "labels": det_labels[i],
                    "scores": scores[i],
                }
                if angles[i] is not None:
                    entry["angles"] = angles[i]
                result.append(entry)

        return result, losses