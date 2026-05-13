"""
infer_13.py  –  Section 1.3: Analysis & Evaluation
====================================================
Loads a trained OBB checkpoint and produces the full set of deliverables:

  1. mAP at multiple IoU thresholds (0.25, 0.50, 0.75)
  2. Per-class AP table
  3. Precision-Recall curve plot  (saved as PNG)
  4. Qualitative overlays on 10 validation images (GT green vs Pred red)
  5. Comparative table across multiple checkpoints (if --compare_ckpts given)

Usage
-----
# Single model evaluation
python infer_13.py --config config/st.yaml \\
                   --ckpt st/tv_frcnn_r50fpn_obb_regression_faster_rcnn_st.pth \\
                   --angle_mode regression

# Compare two checkpoints
python infer_13.py --config config/st.yaml \\
    --compare_ckpts \\
        st/tv_frcnn_r50fpn_obb_regression_faster_rcnn_st.pth,regression,none \\
        st/tv_frcnn_r50fpn_obb_multibin_faster_rcnn_st.pth,multibin,6

All outputs go to  eval_13/
"""

import argparse
import os

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from dataset.st import SceneTextDataset
import detection
from detection.faster_rcnn import FastRCNNPredictor
from detection.roi_heads import RoIHeads
from train_12 import (AngleRegressorHead, AngleMultiBinHead,
                       build_model, rotated_iou, _draw_obb)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ═══════════════════════════════════════════════════════════════════════════════
# Collect raw detections and ground truths
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def collect_preds_gts(model, val_loader, dataset):
    model.eval()
    preds, gts = [], []

    for ims, targets, _ in tqdm(val_loader, desc='Collecting predictions'):
        ims = [im.float().to(device) for im in ims]
        outputs = model(ims, None)

        for output, target in zip(outputs, targets):
            pred_dict = {lbl: [] for lbl in dataset.label2idx}
            boxes  = output['boxes'].cpu().numpy()
            labels = output['labels'].cpu().numpy()
            scores = output['scores'].cpu().numpy()
            angles = output.get('angles', None)
            if angles is not None:
                angles = angles.cpu().numpy()

            for j in range(len(boxes)):
                ang   = float(angles[j]) if angles is not None else 0.0
                lname = dataset.idx2label.get(int(labels[j]), 'text')
                pred_dict[lname].append([*boxes[j], float(scores[j]), ang])
            preds.append(pred_dict)

            gt_dict   = {lbl: [] for lbl in dataset.label2idx}
            gt_boxes  = target['bboxes'].numpy()
            gt_labels = target['labels'].numpy()
            gt_angles = target['angles'].numpy()
            for j in range(len(gt_boxes)):
                lname = dataset.idx2label.get(int(gt_labels[j]), 'text')
                gt_dict[lname].append([*gt_boxes[j], float(gt_angles[j])])
            gts.append(gt_dict)

    return preds, gts


# ═══════════════════════════════════════════════════════════════════════════════
# mAP at a single threshold (returns per-class recall/precision arrays too)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ap_for_label(label, det_boxes, gt_boxes, iou_threshold):
    cls_dets = [
        (im_idx, entry)
        for im_idx, im_dets in enumerate(det_boxes)
        if label in im_dets
        for entry in im_dets[label]
    ]
    cls_dets = sorted(cls_dets, key=lambda k: -k[1][4])

    gt_matched = [[False] * len(im.get(label, [])) for im in gt_boxes]
    num_gts    = sum(len(im.get(label, [])) for im in gt_boxes)

    tp = [0] * len(cls_dets)
    fp = [0] * len(cls_dets)

    for det_idx, (im_idx, det) in enumerate(cls_dets):
        im_gts = gt_boxes[im_idx].get(label, [])
        best_iou, best_idx = -1, -1
        for gt_idx, gt in enumerate(im_gts):
            iou = rotated_iou(det[:4], det[5], gt[:4], gt[4])
            if iou > best_iou:
                best_iou, best_idx = iou, gt_idx

        if best_iou < iou_threshold or (best_idx >= 0 and gt_matched[im_idx][best_idx]):
            fp[det_idx] = 1
        else:
            tp[det_idx] = 1
            if best_idx >= 0:
                gt_matched[im_idx][best_idx] = True

    tp_c = np.cumsum(tp)
    fp_c = np.cumsum(fp)
    eps  = np.finfo(np.float32).eps
    recalls    = tp_c / max(num_gts, eps)
    precisions = tp_c / np.maximum(tp_c + fp_c, eps)

    # 11-pt interpolated AP
    ap = 0.0
    for t in np.arange(0, 1.01, 0.1):
        p = precisions[recalls >= t]
        ap += p.max() if p.size > 0 else 0.0
    ap /= 11.0

    return ap, recalls, precisions, num_gts


def compute_map_at_threshold(det_boxes, gt_boxes, iou_threshold):
    gt_labels = sorted({cls for im in gt_boxes for cls in im})
    all_aps, all_rec, all_prec = {}, {}, {}
    aps = []
    for label in gt_labels:
        ap, rec, prec, num_gts = compute_ap_for_label(
            label, det_boxes, gt_boxes, iou_threshold)
        if num_gts > 0:
            aps.append(ap)
            all_aps[label]  = ap
            all_rec[label]  = rec
            all_prec[label] = prec
        else:
            all_aps[label]  = np.nan
    mean_ap = float(np.mean(aps)) if aps else 0.0
    return mean_ap, all_aps, all_rec, all_prec


# ═══════════════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_pr_curve(all_rec, all_prec, title, save_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for label in sorted(all_rec):
        rec  = np.concatenate([[0.0], all_rec[label],  [1.0]])
        prec = np.concatenate([[0.0], all_prec[label], [0.0]])
        # Precision envelope
        for i in range(len(prec) - 1, 0, -1):
            prec[i - 1] = max(prec[i - 1], prec[i])
        ax.step(rec, prec, where='post', label=label)

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f'Saved {save_path}')


# ═══════════════════════════════════════════════════════════════════════════════
# Qualitative: GT (green) vs Pred (red), 10 images
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def qualitative_analysis(model, dataset, angle_mode, save_dir, num=10):
    import random
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    indices = random.sample(range(len(dataset)), min(num, len(dataset)))

    for k, idx in enumerate(indices):
        im_tensor, target, im_path = dataset[idx]
        img = cv2.imread(im_path)
        if img is None:
            img_np = (im_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Ground truth – green
        for box, ang in zip(target['bboxes'].numpy(), target['angles'].numpy()):
            x1, y1, x2, y2 = box
            _draw_obb(img, (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1, ang,
                      (0, 220, 0), 2)

        # Predictions – red
        output = model([im_tensor.float().to(device)], None)[0]
        boxes  = output['boxes'].cpu().numpy()
        scores = output['scores'].cpu().numpy()
        angles_out = output.get('angles', None)
        if angles_out is not None:
            angles_out = angles_out.cpu().numpy()

        for j in range(len(boxes)):
            x1, y1, x2, y2 = boxes[j]
            ang = float(angles_out[j]) if angles_out is not None else 0.0
            _draw_obb(img, (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1, ang,
                      (0, 60, 255), 2)
            cv2.putText(img, f'{scores[j]:.2f} {ang:.0f}°',
                        (int(x1), max(int(y1)-4, 14)),
                        cv2.FONT_HERSHEY_PLAIN, 0.85, (0, 220, 255), 1)

        # Legend
        cv2.rectangle(img, (0, 0), (220, 22), (0, 0, 0), -1)
        cv2.putText(img, 'GT=green  Pred=red', (4, 15),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

        cv2.imwrite(os.path.join(save_dir, f'qual_{k:02d}.png'), img)

    print(f'Qualitative images saved to {save_dir}')


# ═══════════════════════════════════════════════════════════════════════════════
# Main evaluation routine
# ═══════════════════════════════════════════════════════════════════════════════

def _infer_num_bins_from_ckpt(ckpt_path: str, angle_mode: str) -> int:
    """
    Read the checkpoint and infer num_bins from the angle predictor weight shape.
    Works for multibin (shape[0] == num_bins) and regression (shape[0] == 1).
    Falls back to 6 if the key is absent.
    """
    sd = torch.load(ckpt_path, map_location='cpu')
    key = 'roi_heads.angle_predictor.fc.weight'
    if key not in sd:
        return 6          # no angle predictor — default
    shape = sd[key].shape[0]
    if angle_mode == 'regression':
        return 6          # shape[0]==1 for regression; num_bins irrelevant
    return int(shape)     # for multibin this is exactly num_bins


def evaluate_single(config_path, ckpt_path, angle_mode, num_bins,
                    save_dir, label='model'):
    with open(config_path) as fh:
        config = yaml.safe_load(fh)
    dc = config['dataset_params']

    st_val  = SceneTextDataset('test', root_dir=dc['root_dir'])
    val_loader = DataLoader(st_val, batch_size=1, shuffle=False,
                            num_workers=0, collate_fn=lambda x: tuple(zip(*x)))

    # Auto-detect num_bins from the actual checkpoint so we never get a
    # size mismatch even if the caller passes the wrong value.
    detected_bins = _infer_num_bins_from_ckpt(ckpt_path, angle_mode)
    if detected_bins != num_bins:
        print(f'[INFO] num_bins override: caller said {num_bins}, '
              f'checkpoint has {detected_bins} → using {detected_bins}')
        num_bins = detected_bins

    model = build_model(dc['num_classes'], angle_mode, num_bins,
                        angle_loss_weight=1.0)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)

    preds, gts = collect_preds_gts(model, val_loader, st_val)
    os.makedirs(save_dir, exist_ok=True)

    thresholds = [0.25, 0.50, 0.75]
    print(f'\n{"="*60}')
    print(f'  {label}')
    print(f'{"="*60}')

    rows = []
    for thr in thresholds:
        mean_ap, all_aps, all_rec, all_prec = compute_map_at_threshold(
            preds, gts, thr)
        print(f'\n  IoU threshold = {thr:.2f}  |  mAP = {mean_ap:.4f}')
        for cls, ap in sorted(all_aps.items()):
            print(f'    {cls:12s}  AP = {ap:.4f}')
        rows.append((thr, mean_ap, all_aps))

        # P-R curve
        if thr == 0.50:
            pr_path = os.path.join(save_dir, f'pr_curve_{label.replace(" ","_")}.png')
            plot_pr_curve(all_rec, all_prec,
                          f'Precision-Recall  {label}  IoU={thr}',
                          pr_path)

    # Qualitative
    qual_dir = os.path.join(save_dir, f'qual_{label.replace(" ", "_")}')
    qualitative_analysis(model, st_val, angle_mode, qual_dir, num=10)

    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Section 1.3 – OBB evaluation')
    ap.add_argument('--config',      dest='config_path',
                    default='config/st.yaml')
    ap.add_argument('--ckpt',        default=None,
                    help='Single checkpoint to evaluate')
    ap.add_argument('--angle_mode',  default='regression',
                    choices=['regression', 'multibin'])
    ap.add_argument('--num_bins',    type=int, default=6)
    ap.add_argument('--out_dir',     default='eval_13')
    ap.add_argument('--compare_ckpts', nargs='+', default=None,
                    help='Space-separated list of "ckpt,angle_mode,num_bins" '
                         'triples for comparison table. '
                         'E.g.  st/reg.pth,regression,none  '
                         '      st/mb6.pth,multibin,6')
    args = ap.parse_args()

    if args.compare_ckpts:
        # ── comparison mode ───────────────────────────────────────────────────
        all_results = {}
        for spec in args.compare_ckpts:
            parts = spec.split(',')
            if len(parts) != 3:
                raise ValueError(f'Expected ckpt,angle_mode,num_bins  got: {spec}')
            ckpt_p, amode, nbins_s = parts
            nbins = int(nbins_s) if nbins_s.isdigit() else 6
            label = os.path.basename(ckpt_p).replace('.pth', '')
            rows  = evaluate_single(args.config_path, ckpt_p, amode, nbins,
                                    args.out_dir, label=label)
            all_results[label] = rows

        # Print comparison table
        thresholds = [r[0] for r in list(all_results.values())[0]]
        print('\n\nComparison Table')
        print('-' * 60)
        header = 'Model'.ljust(35) + ''.join(f'mAP@{t}'.ljust(10) for t in thresholds)
        print(header)
        print('-' * 60)
        for lbl, rows in all_results.items():
            line = lbl[:34].ljust(35)
            line += ''.join(f'{r[1]:.4f}'.ljust(10) for r in rows)
            print(line)

    elif args.ckpt:
        # ── single model ──────────────────────────────────────────────────────
        evaluate_single(args.config_path, args.ckpt,
                        args.angle_mode, args.num_bins,
                        args.out_dir,
                        label=os.path.basename(args.ckpt).replace('.pth', ''))
    else:
        print('Provide --ckpt or --compare_ckpts')