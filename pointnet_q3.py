"""
pointnet_q3.py  –  Question 3: PointNet on ModelNet-10
=======================================================
Covers all three sub-questions:

  3.1  Data preprocessing + Vanilla PointNet training
  3.2  Permutation invariance test
  3.3  Critical point extraction, visualisation, robustness experiment

Usage (Kaggle / GPU)
--------------------
# Full pipeline (train + all analyses)
python pointnet_q3.py --data_dir ModelNet-10 --epochs 50

# Skip training, load existing checkpoint
python pointnet_q3.py --data_dir ModelNet-10 --test_only

# Disable wandb
python pointnet_q3.py --data_dir ModelNet-10 --no_wandb

Install deps on Kaggle (run in a notebook cell first):
    !pip install plyfile wandb matplotlib

Architecture (simplified PointNet — no T-Nets)
-----------------------------------------------
Input  [B, 3, N]
  → MLP(3→64→64)           per-point
  → MLP(64→128→1024)        per-point
  → GlobalMaxPool           → [B, 1024]
  → FC(1024→512) ReLU BN Dropout(0.3)
  → FC(512→256)  ReLU BN Dropout(0.3)
  → FC(256→num_classes)
"""

import argparse
import glob
import os
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

# ── optional deps ─────────────────────────────────────────────────────────────
try:
    from plyfile import PlyData
    HAS_PLYFILE = True
except ImportError:
    HAS_PLYFILE = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D   # noqa
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')


# ═══════════════════════════════════════════════════════════════════════════════
# 3.1  Data Preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

CLASSES = ['bathtub', 'bed', 'chair', 'desk', 'dresser',
           'monitor', 'night_stand', 'sofa', 'table', 'toilet']
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)


def read_ply(path: str) -> np.ndarray:
    """
    Read a ModelNet PLY file → float32 array [N, 3].
    Falls back to a pure-Python parser if plyfile is not installed.
    """
    if HAS_PLYFILE:
        ply = PlyData.read(path)
        v   = ply['vertex']
        pts = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float32)
        return pts

    # ── fallback: manual ASCII/binary PLY reader ──────────────────────────────
    pts = []
    with open(path, 'rb') as f:
        # parse header
        num_vertices = 0
        is_binary    = False
        is_big_endian = False
        header_done  = False
        while True:
            line = f.readline().decode('ascii', errors='replace').strip()
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            elif line == 'format binary_big_endian 1.0':
                is_binary, is_big_endian = True, True
            elif line == 'format binary_little_endian 1.0':
                is_binary = True
            elif line == 'end_header':
                header_done = True
                break

        if not header_done:
            raise ValueError(f'Invalid PLY header in {path}')

        if is_binary:
            endian = '>' if is_big_endian else '<'
            dtype  = np.dtype(f'{endian}f4')
            raw    = np.frombuffer(f.read(num_vertices * 12), dtype=dtype)
            pts    = raw.reshape(num_vertices, 3).astype(np.float32)
        else:
            for _ in range(num_vertices):
                vals = f.readline().decode().strip().split()
                pts.append([float(vals[0]), float(vals[1]), float(vals[2])])
            pts = np.array(pts, dtype=np.float32)

    return pts


def preprocess(pts: np.ndarray, num_points: int = 1024) -> np.ndarray:
    """
    1. Sample / pad to exactly num_points
    2. Centre to zero mean
    3. Scale to unit sphere (divide by max distance from origin)
    Returns float32 [num_points, 3]
    """
    n = pts.shape[0]

    # Sample or pad
    if n >= num_points:
        idx = np.random.choice(n, num_points, replace=False)
    else:
        idx = np.concatenate([
            np.arange(n),
            np.random.choice(n, num_points - n, replace=True)
        ])
    pts = pts[idx]

    # Centre
    pts = pts - pts.mean(axis=0, keepdims=True)

    # Scale to unit sphere
    dist = np.max(np.linalg.norm(pts, axis=1))
    if dist > 1e-6:
        pts = pts / dist

    return pts.astype(np.float32)


class ModelNet10Dataset(Dataset):
    """
    Loads all .ply files from ModelNet-10/train or ModelNet-10/test.
    Applies preprocessing (centre + unit sphere) and random point sampling.
    Augmentation (train only): random jitter + random rotation around Y axis.
    """

    def __init__(self, split: str, root_dir: str,
                 num_points: int = 1024, augment: bool = True):
        assert split in ('train', 'test')
        self.num_points = num_points
        self.augment    = augment and (split == 'train')

        self.paths: List[str] = []
        self.labels: List[int] = []

        for cls in CLASSES:
            folder = os.path.join(root_dir, split, cls)
            if not os.path.isdir(folder):
                print(f'[WARN] missing folder: {folder}')
                continue
            for p in glob.glob(os.path.join(folder, '*.ply')):
                self.paths.append(p)
                self.labels.append(CLASS2IDX[cls])

        print(f'[ModelNet10] split={split}  samples={len(self.paths)}  '
              f'classes={NUM_CLASSES}')

    def __len__(self):
        return len(self.paths)

    def _augment(self, pts: np.ndarray) -> np.ndarray:
        """Random Y-axis rotation + Gaussian jitter."""
        # Random rotation around Y axis
        theta  = np.random.uniform(0, 2 * np.pi)
        c, s   = np.cos(theta), np.sin(theta)
        Ry     = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        pts    = pts @ Ry.T

        # Jitter
        pts   += np.random.normal(0, 0.02, pts.shape).astype(np.float32)
        pts    = pts.clip(-1, 1)
        return pts

    def __getitem__(self, idx: int):
        pts   = read_ply(self.paths[idx])
        pts   = preprocess(pts, self.num_points)
        if self.augment:
            pts = self._augment(pts)

        # [3, N] — channels first for 1D Conv
        pts_t = torch.from_numpy(pts.T)                  # [3, N]
        lbl_t = torch.tensor(self.labels[idx], dtype=torch.long)
        return pts_t, lbl_t


# ═══════════════════════════════════════════════════════════════════════════════
# 3.1  Vanilla PointNet (no T-Nets)
# ═══════════════════════════════════════════════════════════════════════════════

class PointNetVanilla(nn.Module):
    """
    Simplified PointNet without Spatial or Feature Transform networks.

    Feature extraction MLP (shared weights via Conv1d):
        3 → 64 → 64 → 128 → 1024
    Global max pooling → [B, 1024]
    Classification MLP:
        1024 → 512 → 256 → num_classes
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3):
        super().__init__()

        # ── per-point MLP (implemented as Conv1d for efficiency) ──────────────
        self.conv1 = nn.Sequential(
            nn.Conv1d(3,   64,  1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64,  64,  1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64,  128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 1024, 1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )

        # ── global max pool: applied in forward ───────────────────────────────

        # ── classification head ───────────────────────────────────────────────
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, N]  →  logits [B, num_classes]"""
        x = self.conv1(x)   # [B, 64,   N]
        x = self.conv2(x)   # [B, 64,   N]
        x = self.conv3(x)   # [B, 128,  N]
        x = self.conv4(x)   # [B, 1024, N]

        # Global max pooling
        x = x.max(dim=2)[0]  # [B, 1024]

        x = self.fc1(x)      # [B, 512]
        x = self.fc2(x)      # [B, 256]
        x = self.fc3(x)      # [B, num_classes]
        return x

    def extract_pre_pool_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns per-point features just BEFORE global max pool.
        Used for critical point extraction in 3.3.
        Returns [B, 1024, N]
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)   # [B, 1024, N]
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# Training utilities
# ═══════════════════════════════════════════════════════════════════════════════

def run_epoch(model, loader, criterion, optimizer, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for pts, labels in tqdm(loader, leave=False,
                                 desc='train' if train else 'val  '):
            pts    = pts.to(device)       # [B, 3, N]
            labels = labels.to(device)    # [B]

            if train:
                optimizer.zero_grad()

            logits = model(pts)
            loss   = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * pts.size(0)
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += pts.size(0)

    return total_loss / total, correct / total


def plot_curves(train_losses, val_losses, train_accs, val_accs, save_dir):
    if not HAS_MPL:
        return
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, train_losses, label='Train Loss')
    ax1.plot(epochs, val_losses,   label='Val Loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves'); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accs, label='Train Acc')
    ax2.plot(epochs, val_accs,   label='Val Acc')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Curves'); ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(save_dir, 'training_curves.png')
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f'Saved {path}')


# ═══════════════════════════════════════════════════════════════════════════════
# 3.2  Permutation Invariance
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def test_permutation_invariance(model, test_loader) -> float:
    """
    For every test batch:
      1. Get predictions on original point order.
      2. Randomly permute point order along N dimension.
      3. Get predictions on permuted cloud.
      4. Count % of samples where the predicted class CHANGES.

    Returns: fraction of samples whose prediction changed (should be ~0).
    """
    model.eval()
    changed = 0
    total   = 0

    for pts, _ in tqdm(test_loader, desc='Permutation test', leave=False):
        pts = pts.to(device)                        # [B, 3, N]
        B, C, N = pts.shape

        # Original predictions
        logits_orig = model(pts)
        preds_orig  = logits_orig.argmax(dim=1)     # [B]

        # Permute N dimension independently per sample
        perm      = torch.stack([torch.randperm(N) for _ in range(B)], dim=0)
        # pts: [B,3,N] → gather along dim=2
        perm_exp  = perm.unsqueeze(1).expand(B, C, N).to(device)
        pts_perm  = torch.gather(pts, 2, perm_exp)

        logits_perm = model(pts_perm)
        preds_perm  = logits_perm.argmax(dim=1)     # [B]

        changed += (preds_orig != preds_perm).sum().item()
        total   += B

    frac = changed / total
    print(f'\n[3.2 Permutation Invariance]')
    print(f'  Samples where prediction changed: {changed}/{total} '
          f'({100*frac:.2f}%)')
    print(f'  Expected: ~0%  (PointNet is permutation invariant by design)')
    return frac


# ═══════════════════════════════════════════════════════════════════════════════
# 3.3  Critical Points
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_critical_points(model, pts_tensor: torch.Tensor) -> np.ndarray:
    """
    For a single point cloud [1, 3, N], return the indices of critical points —
    the unique points that contributed to the 1024-dim global max-pool vector.

    Returns: sorted unique indices array, shape [K] where K ≤ 1024.
    """
    model.eval()
    pts_tensor = pts_tensor.to(device)               # [1, 3, N]
    feats = model.extract_pre_pool_features(pts_tensor)  # [1, 1024, N]

    # For each of the 1024 feature dimensions, find which point had the max
    critical_idx = feats[0].argmax(dim=1).cpu().numpy()  # [1024]
    return np.unique(critical_idx)                        # deduplicated


def visualise_critical_points(model, dataset, num_samples: int = 5,
                               save_dir: str = 'vis_q3'):
    """
    For each sample: side-by-side plot of full point cloud (left) and
    critical points highlighted in red over faint grey cloud (right).
    """
    if not HAS_MPL:
        print('[WARN] matplotlib not available — skipping visualisation')
        return

    os.makedirs(save_dir, exist_ok=True)
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    for k, idx in enumerate(indices):
        pts_t, label = dataset[idx]                      # [3, N], int
        pts_np = pts_t.numpy().T                         # [N, 3]
        class_name = CLASSES[label.item()]

        critical_idx = extract_critical_points(
            model, pts_t.unsqueeze(0))                   # [K]

        fig = plt.figure(figsize=(10, 4))

        # Left: full cloud
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(pts_np[:, 0], pts_np[:, 1], pts_np[:, 2],
                    s=1, c='steelblue', alpha=0.6)
        ax1.set_title(f'Full cloud\n{class_name}  N={len(pts_np)}')
        ax1.set_axis_off()

        # Right: critical points (red) over faint grey cloud
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(pts_np[:, 0], pts_np[:, 1], pts_np[:, 2],
                    s=1, c='lightgrey', alpha=0.2)
        crit_pts = pts_np[critical_idx]
        ax2.scatter(crit_pts[:, 0], crit_pts[:, 1], crit_pts[:, 2],
                    s=8, c='crimson', alpha=0.9, zorder=5)
        ax2.set_title(f'Critical points\n{len(critical_idx)}/1024 unique')
        ax2.set_axis_off()

        fig.suptitle(f'Sample {k+1}: {class_name}', fontsize=12)
        fig.tight_layout()
        path = os.path.join(save_dir, f'critical_{k:02d}_{class_name}.png')
        fig.savefig(path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {path}')


@torch.no_grad()
def robustness_experiment(model, test_dataset, save_dir: str = 'vis_q3'):
    """
    3.3 Robustness:
    1. Extract critical points for every test sample.
    2. Build a sparse cloud from those critical points only.
    3. Run inference on the sparse cloud.
    4. Report accuracy and compare with full-cloud accuracy.
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    full_correct   = 0
    sparse_correct = 0
    total          = 0
    sparse_sizes   = []

    for idx in tqdm(range(len(test_dataset)),
                    desc='Robustness experiment', leave=False):
        pts_t, label = test_dataset[idx]
        label_val    = label.item()

        # ── full cloud accuracy ───────────────────────────────────────────────
        logits_full = model(pts_t.unsqueeze(0).to(device))
        pred_full   = logits_full.argmax(dim=1).item()
        full_correct += int(pred_full == label_val)

        # ── critical points only ──────────────────────────────────────────────
        crit_idx = extract_critical_points(model, pts_t.unsqueeze(0))  # [K]
        sparse_sizes.append(len(crit_idx))

        pts_np   = pts_t.numpy().T                         # [N, 3]
        crit_pts = pts_np[crit_idx]                        # [K, 3]

        # Preprocess the sparse cloud (re-centre + scale)
        crit_pts = crit_pts - crit_pts.mean(axis=0, keepdims=True)
        dist = np.max(np.linalg.norm(crit_pts, axis=1))
        if dist > 1e-6:
            crit_pts = crit_pts / dist

        # Pad to num_points (1024) by repeating
        K = len(crit_pts)
        num_points = pts_t.shape[1]
        if K < num_points:
            pad_idx  = np.random.choice(K, num_points - K, replace=True)
            crit_pts = np.concatenate([crit_pts, crit_pts[pad_idx]], axis=0)
        else:
            crit_pts = crit_pts[:num_points]

        sparse_t = torch.from_numpy(crit_pts.T).unsqueeze(0).to(device)  # [1,3,N]
        logits_sparse = model(sparse_t)
        pred_sparse   = logits_sparse.argmax(dim=1).item()
        sparse_correct += int(pred_sparse == label_val)

        total += 1

    full_acc   = full_correct   / total
    sparse_acc = sparse_correct / total

    print(f'\n[3.3 Robustness Experiment]')
    print(f'  Full cloud accuracy  : {full_acc:.4f}  ({full_correct}/{total})')
    print(f'  Sparse (critical pts): {sparse_acc:.4f}  ({sparse_correct}/{total})')
    print(f'  Avg critical pts/sample: {np.mean(sparse_sizes):.1f} / 1024')
    print(f'\n  Interpretation:')
    print(f'  Because the global feature is formed by max-pooling, only the')
    print(f'  critical points contributed to the representation in the first')
    print(f'  place. Passing only those points should give near-identical')
    print(f'  accuracy — the non-critical points were "invisible" to the model.')

    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(['Full cloud', 'Critical pts only'],
               [full_acc * 100, sparse_acc * 100],
               color=['steelblue', 'crimson'])
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Robustness: full vs critical-points-only')
        ax.set_ylim(0, 100)
        for i, v in enumerate([full_acc, sparse_acc]):
            ax.text(i, v * 100 + 0.5, f'{v*100:.1f}%', ha='center')
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        path = os.path.join(save_dir, 'robustness_bar.png')
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f'  Bar chart saved to {path}')

    return full_acc, sparse_acc


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main(args):
    # ── reproducibility ───────────────────────────────────────────────────────
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # ── wandb ─────────────────────────────────────────────────────────────────
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project,
                       name=f'pointnet_N{args.num_points}_ep{args.epochs}',
                       config=vars(args))
        except ImportError:
            print('[WARN] wandb not installed: pip install wandb')
            use_wandb = False

    # ── datasets ──────────────────────────────────────────────────────────────
    full_train = ModelNet10Dataset('train', args.data_dir,
                                   args.num_points, augment=True)
    test_ds    = ModelNet10Dataset('test',  args.data_dir,
                                   args.num_points, augment=False)

    # 80/20 train-val split
    n_val   = max(1, int(0.2 * len(full_train)))
    n_train = len(full_train) - n_val
    train_ds, val_ds = random_split(
        full_train, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    # ── model ─────────────────────────────────────────────────────────────────
    model = PointNetVanilla(num_classes=NUM_CLASSES,
                            dropout=args.dropout).to(device)
    print(f'Params: {sum(p.numel() for p in model.parameters()):,}')

    ckpt_dir = 'checkpoints_q3'
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'best.pth')

    if args.test_only:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f'Loaded checkpoint from {ckpt_path}')
    else:
        # ── training ──────────────────────────────────────────────────────────
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-5)

        best_val_acc = 0.0
        train_losses, val_losses, train_accs, val_accs = [], [], [], []

        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc = run_epoch(model, train_loader, criterion,
                                         optimizer, train=True)
            vl_loss, vl_acc = run_epoch(model, val_loader,   criterion,
                                         None,      train=False)
            scheduler.step()

            train_losses.append(tr_loss); val_losses.append(vl_loss)
            train_accs.append(tr_acc);    val_accs.append(vl_acc)

            print(f'Ep {epoch:3d}/{args.epochs} | '
                  f'Train loss {tr_loss:.4f} acc {tr_acc:.4f} | '
                  f'Val   loss {vl_loss:.4f} acc {vl_acc:.4f} | '
                  f'lr {scheduler.get_last_lr()[0]:.2e}')

            if use_wandb:
                import wandb
                wandb.log({'epoch': epoch,
                           'train/loss': tr_loss, 'train/acc': tr_acc,
                           'val/loss':   vl_loss, 'val/acc':   vl_acc,
                           'lr': scheduler.get_last_lr()[0]})

            torch.save(model.state_dict(),
                       os.path.join(ckpt_dir, 'last.pth'))
            if vl_acc > best_val_acc:
                best_val_acc = vl_acc
                torch.save(model.state_dict(), ckpt_path)
                print(f'  ↑ New best val acc: {best_val_acc:.4f}')

        plot_curves(train_losses, val_losses, train_accs, val_accs, 'vis_q3')
        if use_wandb:
            import wandb
            wandb.finish()

        print(f'\nBest val acc: {best_val_acc:.4f}')

    # ── 3.1 Test accuracy ─────────────────────────────────────────────────────
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    _, test_acc = run_epoch(model, test_loader, nn.CrossEntropyLoss(),
                             None, train=False)
    print(f'\n[3.1 Test Accuracy] {test_acc:.4f}')

    # ── 3.2 Permutation invariance ────────────────────────────────────────────
    test_permutation_invariance(model, test_loader)

    # ── 3.3 Critical points ───────────────────────────────────────────────────
    # Test dataset without augmentation for deterministic critical points
    test_ds_clean = ModelNet10Dataset('test', args.data_dir,
                                      args.num_points, augment=False)
    visualise_critical_points(model, test_ds_clean,
                               num_samples=5, save_dir='vis_q3')
    robustness_experiment(model, test_ds_clean, save_dir='vis_q3')


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Q3 PointNet on ModelNet-10')
    ap.add_argument('--data_dir',      default='ModelNet-10')
    ap.add_argument('--num_points',    type=int,   default=1024)
    ap.add_argument('--epochs',        type=int,   default=50)
    ap.add_argument('--batch_size',    type=int,   default=32)
    ap.add_argument('--lr',            type=float, default=1e-3)
    ap.add_argument('--dropout',       type=float, default=0.3)
    ap.add_argument('--wandb_project', default='CV_A3_Q3')
    ap.add_argument('--no_wandb',      action='store_true')
    ap.add_argument('--test_only',     action='store_true',
                    help='Skip training, load best.pth and run all analyses')
    args = ap.parse_args()
    main(args)