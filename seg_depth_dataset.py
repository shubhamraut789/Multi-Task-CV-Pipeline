"""
seg_depth_dataset.py  –  Dataset for Question 2 (Segmentation + Depth)
=======================================================================
Dataset structure expected:
    Segmentation&Depth/
    ├── train/
    │   ├── images/   ← RGB  256×256  uint8
    │   ├── depth/    ← grayscale 256×256  uint8  (single-channel or RGB-mode)
    │   └── labels/   ← class-ID map 256×256, values 0-13, stored as RGB PNG
    └── test/
        ├── images/
        ├── depth/
        └── labels/

Label encoding
--------------
The label PNG stores class IDs in the pixel values (0–13). Even though PIL
opens it as RGB, all three channels are identical — we take channel 0.

Depth encoding
--------------
Single-channel or RGB grayscale, uint8, values 0–255.
We normalise to [0, 1] as float32.

Augmentation (train only)
--------------------------
Random horizontal flip applied consistently to image, depth and label.
"""

import glob
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


NUM_CLASSES = 14   # classes 0-13


class SegDepthDataset(Dataset):
    def __init__(self, split: str, root_dir: str, augment: bool = True):
        """
        Parameters
        ----------
        split    : 'train' or 'test'
        root_dir : path to 'Segmentation&Depth' folder
        augment  : apply random horizontal flip (train only)
        """
        assert split in ('train', 'test'), "split must be 'train' or 'test'"
        self.augment  = augment and (split == 'train')
        self.split    = split

        img_dir   = os.path.join(root_dir, split, 'images')
        depth_dir = os.path.join(root_dir, split, 'depth')
        lbl_dir   = os.path.join(root_dir, split, 'labels')

        # Match by stem — assumes identical filenames across folders
        stems = sorted(
            os.path.splitext(os.path.basename(f))[0]
            for f in glob.glob(os.path.join(img_dir, '*.png'))
        )
        # Also accept .jpg images
        stems += sorted(
            os.path.splitext(os.path.basename(f))[0]
            for f in glob.glob(os.path.join(img_dir, '*.jpg'))
        )
        stems = sorted(set(stems))

        self.samples = []
        for stem in stems:
            # Try PNG first, then JPG for image
            img_path = os.path.join(img_dir, stem + '.png')
            if not os.path.exists(img_path):
                img_path = os.path.join(img_dir, stem + '.jpg')

            dep_path = os.path.join(depth_dir, stem + '.png')
            lbl_path = os.path.join(lbl_dir,   stem + '.png')

            if (os.path.exists(img_path) and
                    os.path.exists(dep_path) and
                    os.path.exists(lbl_path)):
                self.samples.append((img_path, dep_path, lbl_path))

        print(f'[SegDepthDataset] split={split}  samples={len(self.samples)}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.
        Args:
            idx (int): Index of the sample to retrieve.
        Returns:
            tuple: A tuple containing:
                - image_t (torch.Tensor): Normalized RGB image tensor of shape [3, H, W].
                  Values are normalized using ImageNet statistics (mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]).
                - depth_t (torch.Tensor): Normalized depth map tensor of shape [1, H, W].
                  Values are normalized to range [0, 1].
                - label_t (torch.Tensor): Semantic segmentation label tensor of shape [H, W]
                  with dtype int64. Class indices are in range [0, NUM_CLASSES-1].
        Notes:
            - Random horizontal flips are applied to all modalities if augmentation is enabled.
            - If the label image has 3 channels, only the red channel is used.
            - Label values are clamped to valid range [0, NUM_CLASSES-1] to handle edge cases.
        """
        img_path, dep_path, lbl_path = self.samples[idx]

        # ── load ──────────────────────────────────────────────────────────────
        image = Image.open(img_path).convert('RGB')           # H×W×3
        depth = Image.open(dep_path).convert('L')             # H×W  (grayscale)
        label = Image.open(lbl_path)                          # H×W or H×W×3

        # ── augmentation ──────────────────────────────────────────────────────
        if self.augment and random.random() > 0.5:
            image = TF.hflip(image)
            depth = TF.hflip(depth)
            label = TF.hflip(label)

        # ── image → float tensor [3, H, W]  normalised to [0,1] ──────────────
        
        image_t = TF.to_tensor(image)
        image_t = TF.normalize(image_t,
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225])                         # [3,256,256] float

        # ── depth → float tensor [1, H, W]  normalised to [0,1] ─────────────
        depth_np = np.array(depth, dtype=np.float32) / 255.0  # [H,W]
        depth_t  = torch.from_numpy(depth_np).unsqueeze(0)    # [1,H,W]

        # ── label → long tensor [H, W]  values in [0, NUM_CLASSES-1] ─────────
        lbl_np = np.array(label)
        if lbl_np.ndim == 3:
            lbl_np = lbl_np[:, :, 0]                          # take R channel
        lbl_np = lbl_np.astype(np.int64)
        # safety clamp — should not be needed but guards against odd encodings
        lbl_np = lbl_np.clip(0, NUM_CLASSES - 1)
        label_t = torch.from_numpy(lbl_np)                    # [H,W]  int64

        return image_t, depth_t, label_t