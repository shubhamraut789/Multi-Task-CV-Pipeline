"""
train_q2.py  –  Question 2: Multi-Task U-Net Training
======================================================
Trains all three U-Net variants:
  vanilla   → 2.1 Vanilla Multi-Task U-Net (with skip connections)
  noskip    → 2.2 Multi-Task U-Net without skip connections
  residual  → 2.3 Multi-Task U-Net with residual blocks

Loss
----
  Combined = seg_weight * CrossEntropy(seg) + depth_weight * RMSE(depth)
  Default: seg_weight=1.0, depth_weight=1.0

Metrics logged every epoch (train + val)
----------------------------------------
  loss_seg, loss_depth, loss_total, mIoU, depth_RMSE

Usage
-----
# Train vanilla (2.1)
python train_q2.py --model vanilla --epochs 30 --wandb_project CV_A3_Q2

# Train no-skip (2.2)  — uses same loss weights from 2.1
python train_q2.py --model noskip --epochs 30 --wandb_project CV_A3_Q2

# Train residual (2.3)
python train_q2.py --model residual --epochs 30 --wandb_project CV_A3_Q2

# Disable wandb (offline / no account)
python train_q2.py --model vanilla --no_wandb

Outputs
-------
  checkpoints/<model>/best.pth        ← best val mIoU checkpoint
  checkpoints/<model>/last.pth        ← last epoch checkpoint
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from seg_depth_dataset import SegDepthDataset, NUM_CLASSES
from unet_models import VanillaMultiTaskUNet, NoSkipMultiTaskUNet, ResidualMultiTaskUNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# ═══════════════════════════════════════════════════════════════════════════════
# Loss functions
# ═══════════════════════════════════════════════════════════════════════════════

class CombinedLoss(nn.Module):
    """
    total = seg_weight * CrossEntropyLoss(seg_logits, labels)
          + depth_weight * RMSE(depth_pred, depth_gt)

    CrossEntropy handles class imbalance well out of the box.
    RMSE is more interpretable for depth than MSE.
    """
    def __init__(self, seg_weight: float = 1.0, depth_weight: float = 1.0,
                 ignore_index: int = -1):
        super().__init__()
        self.seg_weight   = seg_weight
        self.depth_weight = depth_weight
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, seg_logits, depth_pred, seg_labels, depth_gt):
        loss_seg   = self.ce(seg_logits, seg_labels)
        loss_depth = torch.sqrt(
            nn.functional.mse_loss(depth_pred, depth_gt) + 1e-8
        )
        total = self.seg_weight * loss_seg + self.depth_weight * loss_depth
        return total, loss_seg, loss_depth


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_miou(preds: torch.Tensor, labels: torch.Tensor,
                 num_classes: int = NUM_CLASSES) -> float:
    """
    preds  : [B, H, W]  int64  predicted class indices
    labels : [B, H, W]  int64  ground truth class indices
    Returns mean IoU over classes that appear in labels.
    """
    ious = []
    preds_np  = preds.cpu().numpy().flatten()
    labels_np = labels.cpu().numpy().flatten()

    for cls in range(num_classes):
        pred_c  = (preds_np  == cls)
        label_c = (labels_np == cls)
        inter   = (pred_c & label_c).sum()
        union   = (pred_c | label_c).sum()
        if union == 0:
            continue    # class not present — skip
        ious.append(inter / union)

    return float(np.mean(ious)) if ious else 0.0


def compute_rmse(depth_pred: torch.Tensor, depth_gt: torch.Tensor) -> float:
    return float(
        torch.sqrt(nn.functional.mse_loss(depth_pred, depth_gt) + 1e-8).item()
    )


# ═══════════════════════════════════════════════════════════════════════════════
# One epoch helpers
# ═══════════════════════════════════════════════════════════════════════════════

def run_epoch(model, loader, criterion, optimizer, train: bool):
    model.train() if train else model.eval()
    total_loss = seg_loss_acc = depth_loss_acc = 0.0
    all_miou, all_rmse = [], []
    n = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, depths, labels in tqdm(loader, leave=False,
                                            desc='train' if train else 'val  '):
            images = images.to(device)          # [B,3,256,256]
            depths = depths.to(device)          # [B,1,256,256]
            labels = labels.to(device)          # [B,256,256]  int64

            if train:
                optimizer.zero_grad()

            seg_logits, depth_pred = model(images)

            loss, l_seg, l_dep = criterion(seg_logits, depth_pred, labels, depths)

            if train:
                loss.backward()
                optimizer.step()

            bs = images.size(0)
            total_loss     += loss.item()  * bs
            seg_loss_acc   += l_seg.item() * bs
            depth_loss_acc += l_dep.item() * bs
            n              += bs

            # Metrics
            pred_cls = seg_logits.argmax(dim=1)       # [B,H,W]
            all_miou.append(compute_miou(pred_cls, labels))
            all_rmse.append(compute_rmse(depth_pred, depths))

    return {
        'loss':       total_loss     / n,
        'loss_seg':   seg_loss_acc   / n,
        'loss_depth': depth_loss_acc / n,
        'miou':       float(np.mean(all_miou)),
        'rmse':       float(np.mean(all_rmse)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Visualisation (save 10 val samples as side-by-side PNG)
# ═══════════════════════════════════════════════════════════════════════════════

import cv2

# A simple fixed colour palette for 14 classes (BGR)
PALETTE = np.array([
    [128,  64, 128], [244,  35, 232], [ 70,  70,  70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170,  30], [220, 220,   0],
    [107, 142,  35], [152, 251, 152], [ 70, 130, 180], [220,  20,  60],
    [255,   0,   0], [  0,   0, 142],
], dtype=np.uint8)


def _seg_to_colour(seg_hw: np.ndarray) -> np.ndarray:
    """seg_hw: [H,W] int, returns [H,W,3] BGR uint8."""
    out = PALETTE[seg_hw.clip(0, len(PALETTE) - 1)]
    return out


@torch.no_grad()
def save_qualitative(model, dataset, epoch: int, save_dir: str, num: int = 10):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    indices = random.sample(range(len(dataset)), min(num, len(dataset)))

    for k, idx in enumerate(indices):
        image, depth, label = dataset[idx]
        image_b = image.unsqueeze(0).to(device)
        seg_logits, depth_pred = model(image_b)
        pred_cls  = seg_logits.argmax(dim=1)[0].cpu().numpy()   # [H,W]
        pred_dep  = depth_pred[0, 0].cpu().numpy()              # [H,W]

        # Convert image tensor → BGR uint8
        img_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # GT mask colour
        gt_mask  = _seg_to_colour(label.numpy())

        # Predicted mask colour
        pred_mask = _seg_to_colour(pred_cls)

        # GT depth → 3-channel grey
        gt_dep_np  = (depth[0].numpy() * 255).astype(np.uint8)
        gt_dep_bgr = cv2.cvtColor(gt_dep_np, cv2.COLOR_GRAY2BGR)

        # Pred depth → 3-channel grey
        pd_dep_np  = (pred_dep * 255).astype(np.uint8)
        pd_dep_bgr = cv2.cvtColor(pd_dep_np, cv2.COLOR_GRAY2BGR)

        # Stack: [input | gt_mask | pred_mask | gt_depth | pred_depth]
        row = np.concatenate(
            [img_bgr, gt_mask, pred_mask, gt_dep_bgr, pd_dep_bgr], axis=1)

        # Add header labels
        row = cv2.copyMakeBorder(row, 20, 0, 0, 0, cv2.BORDER_CONSTANT, value=0)
        for i, lbl in enumerate(['Input', 'GT Seg', 'Pred Seg',
                                   'GT Depth', 'Pred Depth']):
            cv2.putText(row, lbl, (i * 256 + 5, 15),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

        out = os.path.join(save_dir, f'ep{epoch:03d}_s{k:02d}.png')
        cv2.imwrite(out, row)


# ═══════════════════════════════════════════════════════════════════════════════
# Main training function
# ═══════════════════════════════════════════════════════════════════════════════

def train(args):
    # ── reproducibility ───────────────────────────────────────────────────────
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # ── wandb ─────────────────────────────────────────────────────────────────
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=f'{args.model}_sw{args.seg_weight}_dw{args.depth_weight}',
                config=vars(args),
            )
        except ImportError:
            print('[WARN] wandb not installed. Run: pip install wandb')
            use_wandb = False

    # ── datasets ──────────────────────────────────────────────────────────────
    full_train = SegDepthDataset('train', root_dir=args.data_dir, augment=True)
    test_ds    = SegDepthDataset('test',  root_dir=args.data_dir, augment=False)

    # 75% train, 25% val split (as per assignment: "use validation split 25%")
    n_val   = max(1, int(0.25 * len(full_train)))
    n_train = len(full_train) - n_val
    train_ds, val_ds = random_split(
        full_train, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    print(f'Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}')

    # ── model ─────────────────────────────────────────────────────────────────
    model_map = {
        'vanilla':  VanillaMultiTaskUNet,
        'noskip':   NoSkipMultiTaskUNet,
        'residual': ResidualMultiTaskUNet,
    }
    model = model_map[args.model](num_classes=NUM_CLASSES).to(device)
    print(f'Model: {args.model}  '
          f'Params: {sum(p.numel() for p in model.parameters()):,}')

    # ── loss & optimiser ──────────────────────────────────────────────────────
    criterion = CombinedLoss(seg_weight=args.seg_weight,
                             depth_weight=args.depth_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                  weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    # ── checkpoint dir ────────────────────────────────────────────────────────
    ckpt_dir = os.path.join('checkpoints', args.model)
    os.makedirs(ckpt_dir, exist_ok=True)
    vis_dir  = os.path.join('vis_q2', args.model)

    best_miou = 0.0

    # ── training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, criterion, optimizer, train=True)
        vl = run_epoch(model, val_loader,   criterion, optimizer, train=False)

        scheduler.step(vl['miou'])

        print(
            f"Ep {epoch:3d}/{args.epochs} | "
            f"Train loss {tr['loss']:.4f} seg {tr['loss_seg']:.4f} "
            f"dep {tr['loss_depth']:.4f} mIoU {tr['miou']:.4f} "
            f"RMSE {tr['rmse']:.4f} | "
            f"Val   loss {vl['loss']:.4f} seg {vl['loss_seg']:.4f} "
            f"dep {vl['loss_depth']:.4f} mIoU {vl['miou']:.4f} "
            f"RMSE {vl['rmse']:.4f}"
        )

        if use_wandb:
            import wandb
            wandb.log({
                'epoch': epoch,
                'train/loss':       tr['loss'],
                'train/loss_seg':   tr['loss_seg'],
                'train/loss_depth': tr['loss_depth'],
                'train/mIoU':       tr['miou'],
                'train/RMSE':       tr['rmse'],
                'val/loss':         vl['loss'],
                'val/loss_seg':     vl['loss_seg'],
                'val/loss_depth':   vl['loss_depth'],
                'val/mIoU':         vl['miou'],
                'val/RMSE':         vl['rmse'],
                'lr': optimizer.param_groups[0]['lr'],
            })

        # ── checkpoint ────────────────────────────────────────────────────────
        torch.save(model.state_dict(), os.path.join(ckpt_dir, 'last.pth'))
        if vl['miou'] > best_miou:
            best_miou = vl['miou']
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best.pth'))
            print(f'  ↑ New best mIoU: {best_miou:.4f}')

        # ── qualitative (every 5 epochs) ──────────────────────────────────────
        if epoch % 5 == 0 or epoch == args.epochs:
            save_qualitative(model, val_ds, epoch, vis_dir, num=10)

    if use_wandb:
        import wandb
        wandb.finish()

    print(f'\nTraining done. Best val mIoU: {best_miou:.4f}')
    print(f'Checkpoints in: {ckpt_dir}')


# ═══════════════════════════════════════════════════════════════════════════════
# Test-set evaluation (call after training)
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_test(args):
    model_map = {
        'vanilla':  VanillaMultiTaskUNet,
        'noskip':   NoSkipMultiTaskUNet,
        'residual': ResidualMultiTaskUNet,
    }
    model = model_map[args.model](num_classes=NUM_CLASSES).to(device)
    ckpt  = os.path.join('checkpoints', args.model, 'best.pth')
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    test_ds = SegDepthDataset('test', root_dir=args.data_dir, augment=False)
    loader  = DataLoader(test_ds, batch_size=args.batch_size,
                         shuffle=False, num_workers=4)
    criterion = CombinedLoss(args.seg_weight, args.depth_weight)

    metrics = run_epoch(model, loader, criterion, None, train=False)
    print(f'\n[TEST]  model={args.model}')
    print(f'  mIoU  : {metrics["miou"]:.4f}')
    print(f'  RMSE  : {metrics["rmse"]:.4f}')
    print(f'  loss  : {metrics["loss"]:.4f}')

    vis_dir = os.path.join('vis_q2', args.model + '_test')
    save_qualitative(model, test_ds, epoch=0, save_dir=vis_dir, num=10)
    print(f'  Qualitative predictions saved to {vis_dir}')
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Q2 Multi-Task U-Net Training')
    ap.add_argument('--model',          default='vanilla',
                    choices=['vanilla', 'noskip', 'residual'])
    ap.add_argument('--data_dir',       default='Segmentation&Depth')
    ap.add_argument('--epochs',         type=int,   default=30)
    ap.add_argument('--batch_size',     type=int,   default=8)
    ap.add_argument('--lr',             type=float, default=1e-3)
    ap.add_argument('--seg_weight',     type=float, default=1.0,
                    help='Weight for segmentation CE loss')
    ap.add_argument('--depth_weight',   type=float, default=1.0,
                    help='Weight for depth RMSE loss')
    ap.add_argument('--wandb_project',  default='CV_A3_Q2')
    ap.add_argument('--no_wandb',       action='store_true',
                    help='Disable wandb logging')
    ap.add_argument('--test_only',      action='store_true',
                    help='Skip training and run test-set eval on best.pth')
    args = ap.parse_args()

    if args.test_only:
        evaluate_test(args)
    else:
        train(args)
        evaluate_test(args)