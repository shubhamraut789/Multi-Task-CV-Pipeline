"""
dataset/st.py  –  SceneTextDataset with angle support for OBB prediction.

Changes vs original:
  • __getitem__ now reads `theta` from each annotation and normalises it to
    the range [-90, 90) degrees (standard convention for oriented boxes).
  • targets dict gains an 'angles' key: FloatTensor [N].
  • visualise_obb() helper draws oriented boxes on an image for inspection.
"""

import glob
import json
import os

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data.dataset import Dataset


def _normalise_angle(theta_deg: float) -> float:
    """
    Map any angle in degrees to [-90, 90).
    The annotation thetas can be in [0, 360) or [−180, 180) depending on the
    labelling tool, so we normalise uniformly.
    """
    theta = theta_deg % 180.0     # → [0, 180)
    if theta >= 90.0:
        theta -= 180.0            # → [-90, 90)
    return theta


def _obb_corners(xc, yc, w, h, theta_deg):
    """
    Return the 4 corner points of an oriented bounding box as a numpy array
    of shape (4, 2) in image coordinates.
    theta_deg follows OpenCV convention: positive = counter-clockwise.
    """
    rect = ((xc, yc), (w, h), theta_deg)
    box  = cv2.boxPoints(rect)          # (4, 2) float32
    return box.astype(np.float32)


class SceneTextDataset(Dataset):
    def __init__(self, split, root_dir):
        self.split    = split
        self.root_dir = root_dir
        self.im_dir   = os.path.join(root_dir, 'img')
        self.ann_dir  = os.path.join(root_dir, 'annots')

        classes = sorted(['text'])
        classes = ['background'] + classes
        self.label2idx = {c: i for i, c in enumerate(classes)}
        self.idx2label = {i: c for i, c in enumerate(classes)}
        print(self.idx2label)

        # Collect only images that have a matching annotation
        all_images = sorted(glob.glob(os.path.join(self.im_dir, '*.jpg')))
        self.images = [
            im for im in all_images
            if os.path.exists(
                os.path.join(self.ann_dir, os.path.basename(im) + '.json')
            )
        ]
        self.annotations = [
            os.path.join(self.ann_dir, os.path.basename(im) + '.json')
            for im in self.images
        ]

        n = len(self.images)
        split_idx = int(0.8 * n)
        if split == 'train':
            self.images      = self.images[:split_idx]
            self.annotations = self.annotations[:split_idx]
        else:
            self.images      = self.images[split_idx:]
            self.annotations = self.annotations[split_idx:]

    def __len__(self):
        return len(self.images)

    @staticmethod
    def convert_xcycwh_to_xyxy(box):
        x, y, w, h = box
        return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]

    def __getitem__(self, index):
        im_path  = self.images[index]
        im       = Image.open(im_path).convert('RGB')
        im_tensor = torchvision.transforms.ToTensor()(im)

        ann_path = self.annotations[index]
        with open(ann_path, 'r') as f:
            im_info = json.load(f)

        boxes  = []
        angles = []
        for obj in im_info['objects']:
            obb = obj['obb']
            xc, yc, w, h = obb['xc'], obb['yc'], obb['w'], obb['h']
            theta = _normalise_angle(obb.get('theta', 0.0))
            boxes.append(self.convert_xcycwh_to_xyxy([xc, yc, w, h]))
            angles.append(theta)

        targets = {
            'bboxes': torch.as_tensor(boxes,  dtype=torch.float32),
            'labels': torch.ones(len(boxes),   dtype=torch.long),
            'angles': torch.as_tensor(angles,  dtype=torch.float32),
        }
        return im_tensor, targets, im_path


# ═══════════════════════════════════════════════════════════════════════════════
# Visualisation helper
# ═══════════════════════════════════════════════════════════════════════════════

def visualise_obb(dataset, indices=None, num=6, save_dir='obb_vis'):
    """
    Draw oriented bounding boxes (green) and axis-aligned enclosing boxes
    (blue, dashed) for a handful of dataset samples.

    Parameters
    ----------
    dataset : SceneTextDataset instance
    indices : list of int, or None (random)
    num     : how many images to visualise if indices is None
    save_dir: where to save the PNG files
    """
    import random
    os.makedirs(save_dir, exist_ok=True)

    if indices is None:
        indices = random.sample(range(len(dataset)), min(num, len(dataset)))

    for idx in indices:
        im_tensor, targets, im_path = dataset[idx]

        # BGR numpy for OpenCV
        img = cv2.imread(im_path)
        if img is None:
            img_np = (im_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img    = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        bboxes = targets['bboxes'].numpy()     # [N, 4]  xyxy
        angles = targets['angles'].numpy()     # [N]

        for i, (box, theta) in enumerate(zip(bboxes, angles)):
            x1, y1, x2, y2 = box
            xc = (x1 + x2) / 2
            yc = (y1 + y2) / 2
            w  = x2 - x1
            h  = y2 - y1

            # Oriented box corners
            corners = _obb_corners(xc, yc, w, h, theta)
            corners_int = corners.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(img, [corners_int], True, (0, 255, 0), 2)

            # Axis-aligned box (blue)
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                          (255, 100, 0), 1)

            # Angle label
            cv2.putText(img, f'{theta:.1f}°',
                        (int(x1), max(int(y1) - 4, 12)),
                        cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 255), 1)

        fname = os.path.splitext(os.path.basename(im_path))[0]
        out   = os.path.join(save_dir, f'obb_{fname}.png')
        cv2.imwrite(out, img)
        print(f'Saved {out}')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='CV S26 A3 Q1 Dataset')
    ap.add_argument('--num',  type=int, default=8)
    ap.add_argument('--out',  default='obb_vis')
    args = ap.parse_args()

    ds = SceneTextDataset('train', root_dir=args.root)
    visualise_obb(ds, num=args.num, save_dir=args.out)
    print('Done. Check', args.out)