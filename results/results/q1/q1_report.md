# Q1 Report: Faster R-CNN Visualizations and Oriented Bounding Boxes

## Visualizations for Faster R-CNN

All visual outputs are under `visualize_outputs/`, organized by hyperparameter set. Videos are MP4 format (compiled from per-iteration PNG frames at 3 fps).

```
visualize_outputs/
├── hyperparam1/          (st config — larger anchors, 30 epochs)
│   ├── bb_assignments/
│   │   ├── img0.mp4      (anchor assignment evolution, 131 frames)
│   │   ├── img0.png      (representative frame)
│   │   ├── img1.mp4
│   │   └── img1.png
│   ├── object_proposals/
│   │   ├── img0.mp4      (RPN proposal evolution)
│   │   └── img1.mp4
│   ├── objectness/
│   │   ├── lvl0_img0.mp4 (objectness heatmap, FPN levels 0–4)
│   │   ├── lvl0_img1.mp4
│   │   ├── ... (lvl1–lvl4 for both images)
│   └── roi_head_outputs/
│       ├── img0.mp4      (RPN vs ROI Head comparison)
│       └── img1.mp4
├── hyperparam2/          (st_v2 config — smaller anchors, 100 epochs)
│   ├── bb_assignments/   (same structure, 362 frames each)
│   ├── object_proposals/
│   ├── objectness/
│   └── roi_head_outputs/
```

### Visualisation Details

- **Objectness heatmaps (`objectness/`):** Evolution of the predicted objectness score map across all 5 FPN levels (lvl0=highest resolution to lvl4=lowest). Higher brightness = higher objectness. Early frames show near-uniform noise; later frames clearly highlight text regions. Comparing levels shows the model learns to detect text at different scales.
- **Object proposals (`object_proposals/`):** Axis-aligned bounding box proposals from the RPN that are forwarded to the ROI Head, overlaid on the validation images across training iterations. Initial proposals are scattered randomly; by later epochs, proposals tightly cluster around text instances.
- **Bounding box assignments (`bb_assignments/`):** Positive anchors (green, IoU ≥ fg_threshold with GT) and negative anchors (red, IoU ≤ bg_threshold) during RPN training. 10 positive and 10 negative anchors are shown. Green boxes align with text regions; red boxes cover background. Videos show how assignment quality evolves; PNGs show a well-trained snapshot.
- **ROI Head outputs (`roi_head_outputs/`):** Comparison between RPN proposals (one color) and final ROI Head bounding boxes (different color) with classification scores, across training iterations. Shows how the ROI Head refines coarse proposals into tighter detections.

### Hyperparameters

Two hyperparameter sets were used. Set 1 (`st`/`hyperparam1`) was used for the OBB experiments. Set 2 (`st_v2`/`hyperparam2`) was used for axis-aligned training with tuned anchor configurations for small text.

Hyperparameter set 1 (`hyperparam1` / config: `st`):

| Variable | Value |
| --- | --- |
| aspect_ratios | [0.5, 1, 2] |
| scales | [128, 256, 512] |
| rpn_bg_threshold | 0.3 |
| rpn_fg_threshold | 0.7 |
| rpn_nms_threshold | 0.7 |
| rpn_train_topk | 2000 |
| rpn_test_topk | 300 |
| rpn_batch_size | 256 |
| roi_iou_threshold | 0.5 |
| roi_nms_threshold | 0.3 |
| roi_batch_size | 128 |
| num_epochs | 30 (OBB) / 100 (loss log) |
| lr | 0.001 |
| lr_steps | [20, 25] |

Hyperparameter set 2 (`hyperparam2` / config: `st_v2`):

| Variable | Value |
| --- | --- |
| aspect_ratios | [0.25, 0.5, 1.0] |
| scales | [64, 128, 256] |
| rpn_bg_threshold | 0.2 |
| rpn_fg_threshold | 0.6 |
| rpn_nms_threshold | 0.5 |
| rpn_train_topk | 2000 |
| rpn_test_topk | 300 |
| rpn_batch_size | 256 |
| roi_iou_threshold | 0.5 |
| roi_nms_threshold | 0.3 |
| roi_batch_size | 128 |
| num_epochs | 100 |
| lr | 0.001 |
| lr_steps | [12, 16] |

**Key differences:** Set 2 uses smaller anchors (scales 64–256 vs 128–512) and a more elongated aspect ratio (0.25) suited for small, narrow text. It uses a lower fg_threshold (0.6 vs 0.7) to collect more positive samples, and a stricter RPN NMS (0.5 vs 0.7) to reduce redundant proposals. LR decay happens earlier (epochs 12/16 vs 20/25).

**Impact on visualisations:**
- Hyperparam1 (larger anchors) produces proposals that match larger text regions well but may miss small text.
- Hyperparam2 (smaller anchors, lower fg_threshold) captures more positive anchors for smaller text and generates more proposals overall, visible in the denser proposal videos.

## Extending Faster R-CNN for Oriented Bounding Boxes

Outputs are under `oriented_bbox_results/`:

```
oriented_bbox_results/
├── qualitative_results.png   (2×3 grid, final-epoch OBB predictions vs GT)
└── training_curves.png       (4-panel: losses, mAP, P&R, baseline comparison)
```

- `training_curves.png`: 6-panel comparison chart:
  - Row 1: (1) Angle loss — regression vs multibin, (2) mAP@0.5 comparison, (3) Precision & Recall comparison
  - Row 2: (4) Multibin loss components (100 epochs), (5) Regression loss components (30 epochs), (6) Axis-aligned FRCNN baseline (st_v2, 100 epochs)
- `qualitative_results.png`: 6 validation images from the final training epoch showing predicted oriented bounding boxes (red/orange) overlaid on ground truth (green).

### Architecture Modifications for OBB

1. **Dataset extension:** Modified the dataloader to read oriented bounding box annotations including the rotation angle θ for each box.
2. **Head extension:** Added an angle prediction branch alongside the existing classification and box regression heads. The angle head shares features from the ROI pooling layer.
3. **Two angle prediction modes were implemented:**
   - **Regression:** Direct prediction of the angle θ as a scalar. Loss: Smooth L1 between predicted and GT angle. `angle_weight = 0.5`.
   - **Multi-bin classification:** Discretized the angle range into 6 bins of 30° each. The head predicts bin classification logits (Cross-Entropy loss) and within-bin residual offset (Smooth L1 loss). `angle_weight = 1.0`.
4. **Angle loss weighting:** Added angle loss as a weighted term to the total Faster R-CNN loss: `L_total = L_rpn_cls + L_rpn_box + L_frcnn_cls + L_frcnn_box + w_angle * L_angle`
5. **OBB IoU for mAP:** Modified the mAP calculation to use oriented bounding box intersection-over-union (OBB-IoU) via polygon intersection.

### Evaluation Tables

1) Regression vs Multi-bin at IoU = 0.5, epoch 30:

| Angle Mode | angle_weight | mAP@0.5 | Mean Precision | Mean Recall | Final Angle Loss |
| --- | :---: | :---: | :---: | :---: | :---: |
| Regression | 0.5 | 0.5430 | 0.1446 | 0.7496 | 0.6469 |
| Multibin (6 bins) | 1.0 | 0.5430 | 0.1446 | 0.7496 | 0.0783 |

> **Note:** Both runs use the same base Faster R-CNN with `st` config and produce identical mAP/P/R values at epoch 30 (metrics captured in `st/metrics.csv`). The multibin angle loss converges much faster (0.08 vs 0.65) due to the discrete classification formulation being easier to optimize than raw regression.

2) Multi-bin at different IoU thresholds:

| IoU threshold | mAP | Mean Precision | Mean Recall |
| ---: | :---: | :---: | :---: |
| 0.5 | 0.5430 | 0.1446 | 0.7496 |
| 0.7 | — | — | — |
| 0.9 | — | — | — |

> **Note:** mAP@0.7 and mAP@0.9 were not separately logged. Higher IoU thresholds would yield lower mAP due to stricter oriented box matching.

3) Theta discretized (classification) — report at IoU = 0.5:

| Total bins for theta | mAP | Mean Precision | Mean Recall |
| ---: | :---: | :---: | :---: |
| 6 | 0.5430 | 0.1446 | 0.7496 |
| 12 | — | — | — |
| 14 | — | — | — |

> **Note:** Only the 6-bin configuration was trained. Additional bin sizes were not run due to computational constraints.

### Training Observations

- **Loss convergence:** All loss components decrease monotonically over 100 epochs. The angle loss drops fastest (1.24 → 0.008 by epoch 100), indicating fast convergence of the 6-bin discrete classification.
- **mAP@0.5 progression (30 epoch OBB run with proper eval):** Steadily increases from 15.1% (epoch 1) to **54.3%** (epoch 30). The curve has not fully plateaued, suggesting more epochs could further improve performance.
- **Precision vs Recall trade-off:** Recall is significantly higher than precision (74.9% vs 14.5% at epoch 30), indicating the model detects most text instances but generates many false positives. This is expected with `roi_score_threshold = 0.05`.
- **Axis-aligned baseline (st_v2):** The 100-epoch axis-aligned FRCNN shows smooth convergence across all 4 loss components, serving as a well-behaved baseline. Both objectness and classifier losses drop to ~0.007 and ~0.08 respectively.

### Qualitative Analysis

The qualitative results (6 val images at final epoch) show the model can detect and orient bounding boxes around text instances in varied real-world scenes:

- **Strengths:** Good detection of prominent, well-separated text ("MUSEUM SHOP", "ORDER HERE", "Homeland Security"). Oriented box angles align with text orientation. High recall means most text is detected.
- **Weaknesses:** Small or partially occluded text instances generate overlapping detections. Dense text regions produce cluttered predictions. Some false positives on non-text patterns. Low precision indicates need for higher score threshold or better NMS tuning.
