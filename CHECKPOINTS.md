# Model Checkpoints — CV Assignment 3
**Student:** Saurav Deshmukh (sauravdeshmukh200@gmail.com)

> These checkpoints exceed GitHub's 100MB file limit and are hosted on Google Drive.
> **Google Drive Link:** [CV_Assignment3_Checkpoints](https://drive.google.com/drive/folders/15SRRS9cCCRWg2U68XycwR570beLdCTZw?usp=sharing)

---

All checkpoints are in the `checkpoints_upload/` folder with unique names, ready to drag-drop to Drive.

## Q1: Faster R-CNN — Oriented Bounding Boxes

| Upload File | Size | Description |
|-------------|------|-------------|
| `q1_frcnn_axisaligned_st.pth` | 159 MB | Axis-aligned FRCNN (hyperparam1/st, 30 epochs) |
| `q1_obb_multibin_6bins.pth` | 159 MB | OBB multibin 6-bins (hyperparam1/st, 100 epochs) |
| `q1_obb_regression.pth` | 159 MB | OBB regression (hyperparam1/st, 30 epochs) |
| `q1_frcnn_axisaligned_st_v2.pth` | 159 MB | Axis-aligned FRCNN (hyperparam2/st_v2, 100 epochs) |

## Q2: Multi-Task U-Net (Segmentation + Depth)

| Upload File | Size | Description |
|-------------|------|-------------|
| `q2_vanilla_unet_best.pth` | 120 MB | Vanilla U-Net (best val, 30 epochs) |
| `q2_noskip_unet_best.pth` | 108 MB | No-Skip U-Net (best val, 20 epochs) |
| `q2_residual_unet_best.pth` | 127 MB | Residual U-Net (best val, 25 epochs) |

## Q3: PointNet (3D Classification)

| Upload File | Size | Description |
|-------------|------|-------------|
| `q3_pointnet_best.pth` | 3.1 MB | PointNet classifier (best val acc 93.36%, 50 epochs) |

---

**Total:** ~995 MB (8 checkpoints)

### How to load
```python
import torch
model.load_state_dict(torch.load("path/to/<checkpoint>.pth", map_location="cpu"))
```
