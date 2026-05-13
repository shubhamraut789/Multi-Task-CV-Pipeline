"""
train_11.py  –  Section 1.1: Understanding Faster RCNN with Visualizations
===========================================================================
Trains a Faster R-CNN (ResNet-50 FPN) on the Scene Text dataset while
producing the four visualisations required by the assignment on a *fixed*
validation batch at every --vis_freq training steps.

Visualisations saved under  <task_name>/vis11/ :
  1. objectness/lvl{L}/img{I}/       – objectness heatmap per FPN level
  2. proposals/img{I}/               – RPN proposals after NMS
  3. anchors/img{I}/                 – pos (green) / neg (red) anchor assignments
  4. roi_compare/img{I}/             – RPN proposals vs ROI final boxes

After training, every leaf directory of frames is compiled into an .mp4 video.

Usage
-----
    python train_11.py --config config/st.yaml --vis_freq 50 --fps 3

Hyperparameter sets
-------------------
Run twice with different yaml configs (e.g. different anchor sizes / NMS
thresholds) to satisfy the "≥2 hyperparameter sets" requirement.
"""

import os
import random
import argparse

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from dataset.st import SceneTextDataset
import detection
from detection.faster_rcnn import FastRCNNPredictor

# ── device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet stats used by the model's transform
_MEAN = torch.tensor([0.485, 0.456, 0.406])
_STD  = torch.tensor([0.229, 0.224, 0.225])


# ═══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ═══════════════════════════════════════════════════════════════════════════════

def collate_fn(data):
    return tuple(zip(*data))


# ═══════════════════════════════════════════════════════════════════════════════
# Hook / patch manager
# ═══════════════════════════════════════════════════════════════════════════════

def _tensor_to_bgr(t: torch.Tensor) -> np.ndarray:
    """Denormalise a C×H×W float tensor → H×W×C BGR uint8 numpy array."""
    t = (t.cpu() * _STD[:, None, None] + _MEAN[:, None, None]).clamp(0, 1)
    rgb = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


class FasterRCNNCapture:
    """
    Attaches lightweight forward hooks and a method patch to Faster RCNN so
    that intermediate tensors can be read from self.buf after each forward pass.

    Captured keys
    -------------
    resized_imgs      list[ndarray]   – denormalised BGR images after transform
    objectness        list[Tensor]    – [B, num_anchors, H_l, W_l] per FPN level
    rpn_proposals     list[Tensor]    – [N_i, 4] after RPN NMS (resized space)
    anchors           list[Tensor]    – all anchors [A, 4] per image
    anchor_labels     list[Tensor]    – 1=pos, 0=neg, -1=ignore  (training only)
    roi_in_proposals  list[Tensor]    – proposals entering ROI head (resized space)
    roi_boxes         list[Tensor]    – [M, 4] final ROI boxes    (resized space)
    roi_scores        list[Tensor]    – [M]    final ROI scores
    roi_labels        list[Tensor]    – [M]    final ROI label ids
    """

    def __init__(self, model: torch.nn.Module):
        self.model  = model
        self.buf: dict = {}
        self._hooks: list = []
        self._setup_hooks()
        self._patch_rpn_assign()

    # ── hook registration ─────────────────────────────────────────────────────

    def _setup_hooks(self):
        m = self.model

        # 1. Transform → capture resized (de-normalised) images per image
        def h_transform(module, inp, out):
            img_list, _ = out
            imgs = []
            for t, sz in zip(img_list.tensors, img_list.image_sizes):
                imgs.append(_tensor_to_bgr(t[:, : sz[0], : sz[1]]))
            self.buf["resized_imgs"] = imgs

        self._hooks.append(m.transform.register_forward_hook(h_transform))

        # 2. RPNHead → objectness logits per FPN level   [B, num_anchors, H, W]
        def h_rpn_head(module, inp, out):
            logits, _ = out
            self.buf["objectness"] = [o.detach().cpu() for o in logits]

        self._hooks.append(m.rpn.head.register_forward_hook(h_rpn_head))

        # 3. RPN → post-NMS proposals  (one tensor per image, resized space)
        def h_rpn(module, inp, out):
            proposals, _ = out
            self.buf["rpn_proposals"] = [p.detach().cpu() for p in proposals]

        self._hooks.append(m.rpn.register_forward_hook(h_rpn))

        # 4. RoIHeads pre-hook → proposals entering the head (resized space)
        #    forward signature: (features, proposals, image_shapes[, targets])
        def h_roi_pre(module, inp):
            self.buf["roi_in_proposals"] = [p.detach().cpu() for p in inp[1]]

        self._hooks.append(m.roi_heads.register_forward_pre_hook(h_roi_pre))

        # 5. RoIHeads → final detections (eval mode only; still in resized space
        #    because GeneralizedRCNN.postprocess runs *after* roi_heads)
        def h_roi(module, inp, out):
            result, _ = out
            self.buf["roi_boxes"]  = [r.get("boxes",  torch.zeros(0, 4)).detach().cpu() for r in result]
            self.buf["roi_scores"] = [r.get("scores", torch.zeros(0)).detach().cpu() for r in result]
            self.buf["roi_labels"] = [r.get("labels", torch.zeros(0, dtype=torch.long)).detach().cpu() for r in result]

        self._hooks.append(m.roi_heads.register_forward_hook(h_roi))

    # ── RPN assign patch ──────────────────────────────────────────────────────

    def _patch_rpn_assign(self):
        """
        Monkey-patch assign_targets_to_anchors to expose all anchors and their
        labels (1=positive, 0=negative, -1=ignored) after each training pass.
        """
        rpn = self.model.rpn
        if not hasattr(rpn, "assign_targets_to_anchors"):
            print("[WARN] RPN.assign_targets_to_anchors not found – "
                  "anchor assignment visualisation will be skipped.")
            return

        _orig = rpn.assign_targets_to_anchors
        buf   = self.buf

        def _patched(anchors, targets):
            labels, matched_gt = _orig(anchors, targets)
            buf["anchor_labels"] = [l.detach().cpu() for l in labels]
            buf["anchors"]       = [a.detach().cpu() for a in anchors]
            return labels, matched_gt

        rpn.assign_targets_to_anchors = _patched

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


# ═══════════════════════════════════════════════════════════════════════════════
# Drawing utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _label(img: np.ndarray, text: str, row: int = 0):
    """Overlay a small white status label near the top-left corner."""
    y = 16 + row * 14
    cv2.putText(img, text, (5, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0),    2, cv2.LINE_AA)
    cv2.putText(img, text, (5, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)


def _make_heatmap(obj_per_anchor: torch.Tensor, target_hw: tuple) -> np.ndarray:
    """
    obj_per_anchor : [num_anchors, H_feat, W_feat]  (single image, single level)
    Returns a BGR heatmap resized to target_hw = (H_img, W_img).
    """
    hm = torch.sigmoid(obj_per_anchor).mean(0).numpy()   # [H_feat, W_feat]
    mn, mx = hm.min(), hm.max()
    hm = ((hm - mn) / (mx - mn + 1e-6) * 255).astype(np.uint8)
    hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    return cv2.resize(hm, (target_hw[1], target_hw[0]))  # cv2 wants (W, H)


def _draw_boxes(canvas: np.ndarray,
                boxes: np.ndarray,
                color: tuple,
                thickness: int = 1,
                labels: list = None):
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = (int(v) for v in b[:4])
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, canvas.shape[1] - 1), min(y2, canvas.shape[0] - 1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        if labels is not None and i < len(labels):
            cv2.putText(canvas, labels[i], (x1, max(y1 - 3, 10)),
                        cv2.FONT_HERSHEY_PLAIN, 0.75, color, 1)


def _save(path: str, img: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)


# ═══════════════════════════════════════════════════════════════════════════════
# Per-step visualisation functions
# ═══════════════════════════════════════════════════════════════════════════════

def save_objectness_frames(cap: FasterRCNNCapture, step: int, root: str):
    """
    1. Objectness heatmaps
    For each (FPN level, image) → one colourmap overlay saved as a frame.
    Call after a *training-mode* forward pass.
    """
    imgs   = cap.buf.get("resized_imgs", [])
    obj_ls = cap.buf.get("objectness",   [])

    for lvl, obj in enumerate(obj_ls):
        # obj : [B, num_anchors, H_feat, W_feat]
        for i, bgr in enumerate(imgs):
            if i >= obj.shape[0]:
                continue
            hm      = _make_heatmap(obj[i], bgr.shape[:2])
            blended = cv2.addWeighted(hm, 0.55, bgr, 0.45, 0)
            _label(blended, f"Objectness | FPN level {lvl} | img {i} | step {step}")
            _save(f"{root}/objectness/lvl{lvl}/img{i}/{step:07d}.png", blended)


def save_proposal_frames(cap: FasterRCNNCapture, step: int, root: str,
                         max_props: int = 50):
    """
    2. RPN proposals after NMS overlaid on the resized image.
    Call after a *training-mode* forward pass.
    """
    imgs  = cap.buf.get("resized_imgs",  [])
    props = cap.buf.get("rpn_proposals", [])

    for i, (bgr, pr) in enumerate(zip(imgs, props)):
        canvas = bgr.copy()
        boxes  = pr.numpy()[:max_props]
        _draw_boxes(canvas, boxes, (0, 165, 255), thickness=1)
        _label(canvas,
               f"RPN proposals n={len(boxes)} (max shown={max_props}) | "
               f"img {i} | step {step}")
        _save(f"{root}/proposals/img{i}/{step:07d}.png", canvas)


def save_anchor_frames(cap: FasterRCNNCapture, step: int, root: str,
                       max_pos: int = 10, max_neg: int = 10):
    """
    3. Positive (green) / negative (red) anchor assignments.
    Labels: 1.0 = positive, 0.0 = negative, -1.0 = ignored.
    Anchors are clipped to the image boundary before drawing.
    Call after a *training-mode* forward pass.
    """
    imgs    = cap.buf.get("resized_imgs",   [])
    anchors = cap.buf.get("anchors",        [])
    labels  = cap.buf.get("anchor_labels",  [])

    if not anchors:
        return                                # patch not active

    for i, (bgr, anch, lbls) in enumerate(zip(imgs, anchors, labels)):
        h, w   = bgr.shape[:2]
        canvas = bgr.copy()
        a_np   = anch.numpy().copy()
        # Clip to image bounds
        a_np[:, [0, 2]] = a_np[:, [0, 2]].clip(0, w - 1)
        a_np[:, [1, 3]] = a_np[:, [1, 3]].clip(0, h - 1)
        l_np   = lbls.numpy()

        pos_idx = np.where(l_np >  0.5)[0]   # label == 1.0
        neg_idx = np.where(l_np == 0.0)[0]   # label == 0.0

        # Random sample to keep at most max_pos / max_neg
        if len(pos_idx) > max_pos:
            pos_idx = np.random.choice(pos_idx, max_pos, replace=False)
        if len(neg_idx) > max_neg:
            neg_idx = np.random.choice(neg_idx, max_neg, replace=False)

        _draw_boxes(canvas, a_np[neg_idx], (0,   0, 255), thickness=1)   # red
        _draw_boxes(canvas, a_np[pos_idx], (0, 255,   0), thickness=2)   # green
        _label(canvas,
               f"Anchors  pos(green)={len(pos_idx)}  neg(red)={len(neg_idx)} | "
               f"img {i} | step {step}")
        _save(f"{root}/anchors/img{i}/{step:07d}.png", canvas)


def save_roi_comparison_frames(cap: FasterRCNNCapture, step: int, root: str,
                                idx2label: dict):
    """
    4. RPN proposals (yellow, thin) vs ROI-Head final boxes (blue, thick).
    Both are in the resized-image coordinate space.
    Call after an *eval-mode* forward pass.
    """
    imgs     = cap.buf.get("resized_imgs",     [])
    roi_in   = cap.buf.get("roi_in_proposals", [])
    roi_out  = cap.buf.get("roi_boxes",        [])
    roi_sc   = cap.buf.get("roi_scores",       [])
    roi_lb   = cap.buf.get("roi_labels",       [])

    for i, bgr in enumerate(imgs):
        canvas = bgr.copy()

        # Yellow – first 30 RPN proposals entering the ROI head
        if i < len(roi_in):
            _draw_boxes(canvas, roi_in[i].numpy()[:30], (0, 220, 220), thickness=1)

        # Blue – final predicted boxes with class + score
        if i < len(roi_out):
            boxes  = roi_out[i].numpy()
            scores = roi_sc[i].numpy()  if i < len(roi_sc)  else []
            labids = roi_lb[i].numpy()  if i < len(roi_lb)  else []
            box_labels = []
            for j in range(len(boxes)):
                sc  = float(scores[j])  if j < len(scores)  else 0.0
                lid = int(labids[j])    if j < len(labids)   else 1
                box_labels.append(f"{idx2label.get(lid, 'txt')}:{sc:.2f}")
            _draw_boxes(canvas, boxes, (255, 60, 60), thickness=2, labels=box_labels)

        _label(canvas, f"Yellow=RPN proposals | Blue=ROI final | img {i} | step {step}")
        _save(f"{root}/roi_compare/img{i}/{step:07d}.png", canvas)


# ═══════════════════════════════════════════════════════════════════════════════
# Video compilation
# ═══════════════════════════════════════════════════════════════════════════════

def compile_all_videos(root: str, fps: int = 3):
    """Walk every leaf directory under root and compile sorted PNGs → MP4."""
    print(f"\nCompiling visualisation videos (fps={fps}) …")
    for dirpath, subdirs, filenames in os.walk(root):
        if subdirs:
            continue                           # not a leaf dir
        pngs = sorted(f for f in filenames if f.endswith(".png"))
        if not pngs:
            continue
        first = cv2.imread(os.path.join(dirpath, pngs[0]))
        if first is None:
            continue
        h, w  = first.shape[:2]
        vpath = dirpath.rstrip(os.sep) + ".mp4"
        wr    = cv2.VideoWriter(vpath, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for p in pngs:
            frame = cv2.imread(os.path.join(dirpath, p))
            if frame is not None:
                wr.write(frame)
        wr.release()
        print(f"  → {vpath}  ({len(pngs)} frames)")


# ═══════════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════════

def _make_val_targets(val_tgts):
    """Convert val batch targets (with 'bboxes' key) to model format."""
    out = []
    for t in val_tgts:
        out.append({
            "boxes":  t["bboxes"].float().to(device),
            "labels": t["labels"].long().to(device),
        })
    return out


def train(args):
    # ── config ────────────────────────────────────────────────────────────────
    with open(args.config_path) as fh:
        config = yaml.safe_load(fh)
    dc = config["dataset_params"]
    tc = config["train_params"]

    torch.manual_seed(tc["seed"])
    np.random.seed(tc["seed"])
    random.seed(tc["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(tc["seed"])

    # ── datasets ──────────────────────────────────────────────────────────────
    st_train = SceneTextDataset("train", root_dir=dc["root_dir"])
    st_val   = SceneTextDataset("test",  root_dir=dc["root_dir"])

    train_loader = DataLoader(st_train, batch_size=4, shuffle=True,
                              num_workers=4, collate_fn=collate_fn)
    val_loader   = DataLoader(st_val,   batch_size=2, shuffle=False,
                              num_workers=0, collate_fn=collate_fn)

    # Fixed validation batch used for all visualisations
    val_ims, val_tgts, val_fnames = next(iter(val_loader))
    print(f"Fixed val batch: {[os.path.basename(f) for f in val_fnames]}")

    # ── model ─────────────────────────────────────────────────────────────────
    model = detection.fasterrcnn_resnet50_fpn(
        pretrained=True, min_size=600, max_size=1000
    )
    model.roi_heads.box_predictor = FastRCNNPredictor(
        model.roi_heads.box_predictor.cls_score.in_features,
        num_classes=dc["num_classes"],
    )
    model.to(device)

    os.makedirs(tc["task_name"], exist_ok=True)
    vis_root = os.path.join(tc["task_name"], "vis11")

    # ── hooks ─────────────────────────────────────────────────────────────────
    cap = FasterRCNNCapture(model)

    # ── optimiser ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=5e-5, momentum=0.9,
    )

    # ── training ──────────────────────────────────────────────────────────────
    global_step = 0

    for epoch in range(tc["num_epochs"]):
        model.train()
        loss_acc = {
            "loss_objectness":  [],
            "loss_rpn_box_reg": [],
            "loss_classifier":  [],
            "loss_box_reg":     [],
        }

        for ims, targets, _ in tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{tc['num_epochs']}"):

            optimizer.zero_grad()
            for t in targets:
                t["boxes"]  = t["bboxes"].float().to(device); del t["bboxes"]
                t["labels"] = t["labels"].long().to(device)
            images = [im.float().to(device) for im in ims]

            losses = model(images, targets)
            total  = sum(losses.values())
            total.backward()
            optimizer.step()

            for k, v in losses.items():
                if k in loss_acc:
                    loss_acc[k].append(v.item())

            global_step += 1

            # ── visualise every vis_freq steps ────────────────────────────────
            if global_step % args.vis_freq == 0:

                # (a) TRAINING-MODE pass on fixed val batch
                #     → objectness / proposals / anchor assignments
                model.train()
                with torch.no_grad():
                    v_ims  = [im.float().to(device) for im in val_ims]
                    v_tgts = _make_val_targets(val_tgts)
                    model(v_ims, v_tgts)

                save_objectness_frames(cap, global_step, vis_root)
                save_proposal_frames(cap, global_step, vis_root)
                save_anchor_frames(cap, global_step, vis_root)

                # (b) EVAL-MODE pass on fixed val batch
                #     → ROI head comparison
                model.eval()
                with torch.no_grad():
                    v_ims = [im.float().to(device) for im in val_ims]
                    model(v_ims, None)

                save_roi_comparison_frames(cap, global_step, vis_root,
                                           st_val.idx2label)

                model.train()   # restore training mode

        # ── checkpoint ────────────────────────────────────────────────────────
        ckpt_path = os.path.join(tc["task_name"],
                                 "tv_frcnn_r50fpn_" + tc["ckpt_name"])
        torch.save(model.state_dict(), ckpt_path)

        print(
            f"Epoch {epoch + 1:3d} | "
            + "  ".join(
                f"{k.replace('loss_', '')}: {np.mean(v):.4f}"
                for k, v in loss_acc.items()
            )
        )

    # ── compile videos ────────────────────────────────────────────────────────
    cap.remove_hooks()
    compile_all_videos(vis_root, fps=args.fps)
    print("Training complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Section 1.1 – Faster RCNN training with visualisations"
    )
    ap.add_argument("--config",   dest="config_path", default="config/st.yaml",
                    help="Path to YAML config file")
    ap.add_argument("--vis_freq", type=int, default=50,
                    help="Save a visualisation frame every N training steps")
    ap.add_argument("--fps",      type=int, default=3,
                    help="Frame rate for compiled MP4 videos")
    args = ap.parse_args()
    train(args)