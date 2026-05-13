"""
train_12.py  –  Section 1.2: Faster R-CNN for Oriented Bounding Boxes
======================================================================
Extends the baseline Faster R-CNN to predict an additional angle per
detection.  Two modes are supported (set via --angle_mode):

    regression  – direct smooth-L1 angle regression
    multibin    – multi-bin cross-entropy classification (discretised angle)

The training loop also logs Precision, Recall, and mAP at every epoch,
computed against the validation split using Rotated-IoU.

Usage
-----
# Direct regression
python train_12.py --config config/st.yaml --angle_mode regression \\
                   --angle_weight 0.5

# Multi-bin classification (e.g. 6 bins of 30 degrees each)
python train_12.py --config config/st.yaml --angle_mode multibin \\
                   --num_bins 6 --angle_weight 1.0

# To try a different number of bins
python train_12.py --config config/st.yaml --angle_mode multibin \\
                   --num_bins 12 --angle_weight 1.0

Output
------
  <task_name>/                     ← weights
  <task_name>/metrics.csv          ← epoch-level P / R / mAP
  <task_name>/vis12/               ← qualitative prediction images
"""

import argparse
import csv
import os
import random

import cv2
import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torchvision.ops import MultiScaleRoIAlign
from tqdm import tqdm

from dataset.st import SceneTextDataset, visualise_obb
import detection
from detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from detection.roi_heads import RoIHeads        # ← our modified version
from detection.anchor_utils import AnchorGenerator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ═══════════════════════════════════════════════════════════════════════════════
# Angle predictor heads
# ═══════════════════════════════════════════════════════════════════════════════

class AngleRegressorHead(nn.Module):
    """Single linear layer: 1024 → 1 (scalar angle in degrees)."""
    def __init__(self, in_channels: int = 1024):
        super().__init__()
        self.fc = nn.Linear(in_channels, 1)
        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)          # [N, 1]


class AngleMultiBinHead(nn.Module):
    """Single linear layer: 1024 → num_bins (bin logits)."""
    def __init__(self, in_channels: int = 1024, num_bins: int = 6):
        super().__init__()
        self.fc = nn.Linear(in_channels, num_bins)
        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)          # [N, num_bins]


# ═══════════════════════════════════════════════════════════════════════════════
# Rotated-IoU helpers  (polygon-based, no C-extension needed)
# ═══════════════════════════════════════════════════════════════════════════════

def _obb_to_poly(cx, cy, w, h, theta_deg):
    """Return (4, 2) float32 polygon corners for one OBB."""
    rect = ((float(cx), float(cy)), (float(w), float(h)), float(theta_deg))
    return cv2.boxPoints(rect).astype(np.float32)


def _poly_iou(p1, p2):
    """Intersection-over-union of two convex polygons via Sutherland-Hodgman."""
    r1 = p1.reshape((-1, 1, 2)).astype(np.float32)
    r2 = p2.reshape((-1, 1, 2)).astype(np.float32)
    inter_type, inter_pts = cv2.intersectConvexConvex(r1, r2)
    if inter_pts is None or len(inter_pts) == 0:
        return 0.0
    inter_area = cv2.contourArea(inter_pts)
    a1 = cv2.contourArea(r1)
    a2 = cv2.contourArea(r2)
    union = a1 + a2 - inter_area + 1e-6
    return float(inter_area / union)


def rotated_iou(det_xyxy, det_angle, gt_xyxy, gt_angle):
    """
    Rotated IoU between one detection and one ground-truth box.

    det_xyxy / gt_xyxy : (x1, y1, x2, y2) axis-aligned enclosing box
    det_angle / gt_angle: angle in degrees (scalar)
    """
    def _to_params(box_xyxy, ang):
        x1, y1, x2, y2 = box_xyxy
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h   = x2 - x1, y2 - y1
        return cx, cy, w, h, ang

    p1 = _obb_to_poly(*_to_params(det_xyxy, det_angle))
    p2 = _obb_to_poly(*_to_params(gt_xyxy,  gt_angle))
    return _poly_iou(p1, p2)


# ═══════════════════════════════════════════════════════════════════════════════
# mAP with Rotated-IoU
# ═══════════════════════════════════════════════════════════════════════════════

def compute_map_obb(det_boxes, gt_boxes, iou_threshold=0.5):
    """
    det_boxes : list of dicts per image
        {'text': [[x1,y1,x2,y2,score,angle], ...], ...}
    gt_boxes  : list of dicts per image
        {'text': [[x1,y1,x2,y2,angle], ...], ...}

    Returns (mean_ap, all_aps, mean_precision, mean_recall)
    """
    gt_labels = {cls for im in gt_boxes for cls in im}
    gt_labels = sorted(gt_labels)
    all_aps, aps, precs_at_end, recs_at_end = {}, [], [], []

    for label in gt_labels:
        cls_dets = [
            (im_idx, entry)
            for im_idx, im_dets in enumerate(det_boxes)
            if label in im_dets
            for entry in im_dets[label]
        ]
        cls_dets = sorted(cls_dets, key=lambda k: -k[1][4])   # sort by score

        gt_matched  = [[False] * len(im[label]) if label in im else []
                       for im in gt_boxes]
        num_gts     = sum(len(im.get(label, [])) for im in gt_boxes)
        tp = [0] * len(cls_dets)
        fp = [0] * len(cls_dets)

        for det_idx, (im_idx, det) in enumerate(cls_dets):
            im_gts = gt_boxes[im_idx].get(label, [])
            best_iou, best_gt_idx = -1, -1

            for gt_idx, gt in enumerate(im_gts):
                iou = rotated_iou(det[:4], det[5], gt[:4], gt[4])
                if iou > best_iou:
                    best_iou, best_gt_idx = iou, gt_idx

            if best_iou < iou_threshold or gt_matched[im_idx][best_gt_idx]:
                fp[det_idx] = 1
            else:
                tp[det_idx] = 1
                gt_matched[im_idx][best_gt_idx] = True

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        eps    = np.finfo(np.float32).eps
        recalls    = tp_cum / max(num_gts, eps)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, eps)

        # 11-point interpolated AP
        ap = 0.0
        for t in np.arange(0, 1.01, 0.1):
            prec_at_t = precisions[recalls >= t]
            ap += prec_at_t.max() if prec_at_t.size > 0 else 0.0
        ap /= 11.0

        if num_gts > 0:
            aps.append(ap)
            all_aps[label] = ap
            precs_at_end.append(float(precisions[-1]) if len(precisions) > 0 else 0.0)
            recs_at_end.append(float(recalls[-1])    if len(recalls)    > 0 else 0.0)
        else:
            all_aps[label] = np.nan

    mean_ap   = float(np.mean(aps))          if aps else 0.0
    mean_prec = float(np.mean(precs_at_end)) if precs_at_end else 0.0
    mean_rec  = float(np.mean(recs_at_end))  if recs_at_end  else 0.0
    return mean_ap, all_aps, mean_prec, mean_rec


# ═══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ═══════════════════════════════════════════════════════════════════════════════

def collate_fn(data):
    return tuple(zip(*data))


# ═══════════════════════════════════════════════════════════════════════════════
# Model builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_model(num_classes: int, angle_mode: str, num_bins: int,
                angle_loss_weight: float) -> nn.Module:
    """
    Build a Faster R-CNN model with a custom RoIHeads that also predicts angles.
    We call detection.fasterrcnn_resnet50_fpn to get the backbone + RPN, then
    replace roi_heads with our extended version.
    """
    base_model = detection.fasterrcnn_resnet50_fpn(
        pretrained=True, min_size=600, max_size=1000
    )

    # Replace box predictor for our num_classes
    in_features = base_model.roi_heads.box_predictor.cls_score.in_features
    base_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Angle predictor head
    if angle_mode == 'regression':
        angle_predictor = AngleRegressorHead(in_channels=1024)
    elif angle_mode == 'multibin':
        angle_predictor = AngleMultiBinHead(in_channels=1024, num_bins=num_bins)
    else:
        angle_predictor = None

    # Build new RoIHeads with angle support
    # (copy hyper-params from the base model's roi_heads)
    rh = base_model.roi_heads
    new_roi_heads = RoIHeads(
        box_roi_pool        = rh.box_roi_pool,
        box_head            = rh.box_head,
        box_predictor       = rh.box_predictor,
        fg_iou_thresh       = rh.proposal_matcher.high_threshold,
        bg_iou_thresh       = rh.proposal_matcher.low_threshold,
        batch_size_per_image= rh.fg_bg_sampler.batch_size_per_image,
        positive_fraction   = rh.fg_bg_sampler.positive_fraction,
        bbox_reg_weights    = None,
        score_thresh        = rh.score_thresh,
        nms_thresh          = rh.nms_thresh,
        detections_per_img  = rh.detections_per_img,
        angle_predictor     = angle_predictor,
        angle_mode          = angle_mode,
        num_bins            = num_bins,
        angle_loss_weight   = angle_loss_weight,
    )
    base_model.roi_heads = new_roi_heads
    return base_model


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation pass (returns detection / gt dicts for mAP)
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, val_loader, dataset, angle_mode, iou_threshold=0.5):
    model.eval()
    preds, gts = [], []

    for ims, targets, _ in tqdm(val_loader, desc='Eval', leave=False):
        ims = [im.float().to(device) for im in ims]
        outputs = model(ims, None)

        for output, target in zip(outputs, targets):
            # ── predictions ───────────────────────────────────────────────────
            pred_dict = {lbl: [] for lbl in dataset.label2idx}
            boxes  = output['boxes'].cpu().numpy()
            labels = output['labels'].cpu().numpy()
            scores = output['scores'].cpu().numpy()
            angles = output.get('angles', None)
            if angles is not None:
                angles = angles.cpu().numpy()

            for j, (b, lbl, sc) in enumerate(zip(boxes, labels, scores)):
                ang = float(angles[j]) if angles is not None else 0.0
                lname = dataset.idx2label.get(int(lbl), 'text')
                pred_dict[lname].append([*b, float(sc), ang])
            preds.append(pred_dict)

            # ── ground truth ──────────────────────────────────────────────────
            gt_dict   = {lbl: [] for lbl in dataset.label2idx}
            gt_boxes  = target['bboxes'].numpy()
            gt_labels = target['labels'].numpy()
            gt_angles = target['angles'].numpy()
            for j, (b, lbl) in enumerate(zip(gt_boxes, gt_labels)):
                ang   = float(gt_angles[j])
                lname = dataset.idx2label.get(int(lbl), 'text')
                gt_dict[lname].append([*b, ang])
            gts.append(gt_dict)

    mean_ap, all_aps, prec, rec = compute_map_obb(preds, gts, iou_threshold)
    return mean_ap, all_aps, prec, rec


# ═══════════════════════════════════════════════════════════════════════════════
# Qualitative visualisation
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_obb(img, cx, cy, w, h, theta_deg, color, thickness=2):
    rect    = ((float(cx), float(cy)), (float(w), float(h)), float(theta_deg))
    corners = cv2.boxPoints(rect).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [corners], True, color, thickness)


@torch.no_grad()
def save_qualitative(model, dataset, epoch: int, save_dir: str,
                     angle_mode, num_samples=6):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    for k, idx in enumerate(indices):
        im_tensor, target, im_path = dataset[idx]
        img_bgr = cv2.imread(im_path)
        if img_bgr is None:
            img_np  = (im_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        canvas = img_bgr.copy()

        # Ground truth (green OBB)
        for box, ang in zip(target['bboxes'].numpy(), target['angles'].numpy()):
            x1, y1, x2, y2 = box
            cx, cy = (x1+x2)/2, (y1+y2)/2
            w, h   = x2-x1, y2-y1
            _draw_obb(canvas, cx, cy, w, h, ang, (0, 220, 0), 2)

        # Predictions (red OBB + score)
        output = model([im_tensor.float().to(device)], None)[0]
        boxes  = output['boxes'].cpu().numpy()
        scores = output['scores'].cpu().numpy()
        angles = output.get('angles', None)
        if angles is not None:
            angles = angles.cpu().numpy()

        for j, (b, sc) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = b
            ang = float(angles[j]) if angles is not None else 0.0
            cx, cy = (x1+x2)/2, (y1+y2)/2
            w, h   = x2-x1, y2-y1
            _draw_obb(canvas, cx, cy, w, h, ang, (0, 60, 255), 2)
            cv2.putText(canvas, f'{sc:.2f}|{ang:.0f}°',
                        (int(x1), max(int(y1)-4, 12)),
                        cv2.FONT_HERSHEY_PLAIN, 0.85, (0, 200, 255), 1)

        cv2.putText(canvas, 'GT=green  Pred=red', (5, 18),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
        out = os.path.join(save_dir, f'ep{epoch:03d}_sample{k}.png')
        cv2.imwrite(out, canvas)


# ═══════════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════════

def train(args):
    # ── config ────────────────────────────────────────────────────────────────
    with open(args.config_path) as fh:
        config = yaml.safe_load(fh)
    dc = config['dataset_params']
    tc = config['train_params']

    torch.manual_seed(tc['seed'])
    np.random.seed(tc['seed'])
    random.seed(tc['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(tc['seed'])

    # ── datasets ──────────────────────────────────────────────────────────────
    st_train = SceneTextDataset('train', root_dir=dc['root_dir'])
    st_val   = SceneTextDataset('test',  root_dir=dc['root_dir'])

    train_loader = DataLoader(st_train, batch_size=4, shuffle=True,
                              num_workers=4, collate_fn=collate_fn)
    val_loader   = DataLoader(st_val,   batch_size=1, shuffle=False,
                              num_workers=0, collate_fn=collate_fn)

    # ── dataset annotation visualisation (once, at start) ────────────────────
    vis_ann_dir = os.path.join(tc['task_name'], 'ann_vis')
    visualise_obb(st_train, num=8, save_dir=vis_ann_dir)
    print(f'Annotation OBB visualisations saved to {vis_ann_dir}')

    # ── model ─────────────────────────────────────────────────────────────────
    model = build_model(
        num_classes      = dc['num_classes'],
        angle_mode       = args.angle_mode,
        num_bins         = args.num_bins,
        angle_loss_weight= args.angle_weight,
    )
    model.to(device)

    os.makedirs(tc['task_name'], exist_ok=True)

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=5e-5, momentum=0.9,
    )

    # ── metrics CSV ───────────────────────────────────────────────────────────
    csv_path = os.path.join(tc['task_name'], 'metrics.csv')
    csv_file = open(csv_path, 'w', newline='')
    writer   = csv.writer(csv_file)
    writer.writerow(['epoch', 'rpn_cls', 'rpn_loc', 'frcnn_cls',
                     'frcnn_loc', 'angle_loss',
                     'mAP@0.5', 'precision', 'recall'])

    # ── training ──────────────────────────────────────────────────────────────
    num_epochs = tc['num_epochs']

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = {k: [] for k in ['loss_objectness', 'loss_rpn_box_reg',
                                         'loss_classifier', 'loss_box_reg',
                                         'loss_angle']}

        for ims, targets, _ in tqdm(train_loader,
                                     desc=f'Epoch {epoch+1}/{num_epochs}'):
            optimizer.zero_grad()
            for t in targets:
                t['boxes']  = t['bboxes'].float().to(device); del t['bboxes']
                t['labels'] = t['labels'].long().to(device)
                t['angles'] = t['angles'].float().to(device)
            images = [im.float().to(device) for im in ims]

            losses = model(images, targets)
            total  = sum(losses.values())
            total.backward()
            optimizer.step()

            for k in epoch_losses:
                if k in losses:
                    epoch_losses[k].append(losses[k].item())

        # ── checkpoint ────────────────────────────────────────────────────────
        ckpt = os.path.join(tc['task_name'],
                            f'tv_frcnn_r50fpn_obb_{args.angle_mode}_'
                            + tc['ckpt_name'])
        torch.save(model.state_dict(), ckpt)

        # ── evaluate ──────────────────────────────────────────────────────────
        mean_ap, all_aps, prec, rec = evaluate(
            model, val_loader, st_val, args.angle_mode)

        # ── log ───────────────────────────────────────────────────────────────
        def _m(k):
            v = epoch_losses[k]
            return float(np.mean(v)) if v else 0.0

        print(
            f"Ep {epoch+1:3d} | "
            f"rpn_cls {_m('loss_objectness'):.4f} "
            f"rpn_loc {_m('loss_rpn_box_reg'):.4f} | "
            f"cls {_m('loss_classifier'):.4f} "
            f"box {_m('loss_box_reg'):.4f} "
            f"ang {_m('loss_angle'):.4f} | "
            f"mAP@0.5 {mean_ap:.4f}  P {prec:.4f}  R {rec:.4f}"
        )
        writer.writerow([
            epoch + 1,
            _m('loss_objectness'), _m('loss_rpn_box_reg'),
            _m('loss_classifier'), _m('loss_box_reg'),
            _m('loss_angle'),
            mean_ap, prec, rec,
        ])
        csv_file.flush()

        # ── qualitative samples ───────────────────────────────────────────────
        vis_dir = os.path.join(tc['task_name'], 'vis12')
        save_qualitative(model, st_val, epoch + 1, vis_dir, args.angle_mode)

        model.train()   # back to train mode

    csv_file.close()
    print(f'Training complete. Metrics saved to {csv_path}')


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Section 1.2 – OBB Faster R-CNN training')
    ap.add_argument('--config',       dest='config_path',
                    default='config/st.yaml')
    ap.add_argument('--angle_mode',   default='regression',
                    choices=['regression', 'multibin'],
                    help='How to predict angle: regression or multibin')
    ap.add_argument('--num_bins',     type=int, default=6,
                    help='Number of angle bins for multibin mode')
    ap.add_argument('--angle_weight', type=float, default=1.0,
                    help='Weight for angle loss term')
    args = ap.parse_args()
    train(args)