# Q3 Report: Learning on Point Clouds (PointNet on ModelNet-10)

## Required WandB Links
- Training run (50 epochs): https://wandb.ai/sauravdeshmukh200-iiit-hyderabad/CV_A3_Q3/runs/h5sery7a
- Combined Report: https://api.wandb.ai/links/sauravdeshmukh200-iiit-hyderabad/vddjynvl

## Folder Map
- losses.png — Training/validation loss and accuracy curves over 50 epochs
- critical_points.png — 5 test samples showing full cloud vs. critical points

## 3.1 Losses and Metrics

### Architecture
Simplified PointNet (no T-Nets):
- **Feature extraction MLP** (shared weights via Conv1d): 3 → 64 → 64 → 128 → 1024
- **Global max pooling** → [B, 1024]
- **Classification MLP**: 1024 → 512 → 256 → 10 (with BatchNorm, ReLU, Dropout=0.3)
- **Parameters:** 805,578

### Training Configuration
- **Dataset:** ModelNet-10 (3,992 train + 908 test samples, 10 classes)
- **Data split:** 80/20 train-val split (3,193 train / 799 val)
- **Preprocessing:** Sample 1024 points per cloud → centre to zero mean → scale to unit sphere
- **Augmentation (train only):** Random Y-axis rotation + Gaussian jitter (σ=0.02)
- **Optimizer:** Adam (lr=1e-3, weight_decay=1e-4)
- **Scheduler:** CosineAnnealingLR (T_max=50, η_min=1e-5)
- **Loss:** CrossEntropyLoss
- **Epochs:** 50, batch_size=32
- **GPU:** Tesla T4 (Kaggle)

### Plots
- losses.png contains 2 panels:
  1. Training and Validation loss (cross-entropy) over 50 epochs
  2. Training and Validation accuracy (%) over 50 epochs

### Final Metrics
- **Best val accuracy:** 0.9336 (epoch 44)
- **Test accuracy:** **0.8998** (89.98%)
- **Final train loss:** 0.1569
- **Final val loss:** 0.2033
- **Final train accuracy:** 0.9441

### Training Observations
- Early training (epochs 1–10): Rapid convergence with some validation oscillation (e.g., epoch 3 val_loss spike to 1.94, val_acc drop to 0.51 — likely a difficult batch or gradient spike).
- Mid training (epochs 10–30): Steady improvement with train/val curves tracking closely. Best val acc = 0.90 reached at epoch 27.
- Late training (epochs 30–50): CosineAnnealing smoothly decays LR. Fine-tuning phase yields best val acc = 0.9336 at epoch 44. Train acc reaches 94.4%.
- Mild overfitting gap (~2%) between train and val accuracy in late epochs, which is expected and well-controlled by dropout and weight decay.

## 3.2 Permutation Invariance

- **Original accuracy:** 0.8998
- **Permuted accuracy** (after random shuffle of input points per sample): 0.8998
- **Samples where prediction changed:** 0/908 (**0.00%**)

**Is the result what you expected?**
Yes, exactly 0% of predictions changed. This is because PointNet achieves permutation invariance *by design*: the feature extraction MLP operates independently on each point (shared Conv1d weights), and the global feature is computed via **max-pooling** over the point dimension. Since max-pooling is a symmetric function (its output does not depend on the order of inputs), permuting the point order has zero effect on the final classification result.

## 3.3 Critical Point Analysis and Robustness

### Critical Point Extraction Method
For each test sample:
1. Extract per-point features from the last Conv1d layer → [1, 1024, N]
2. For each of the 1024 feature channels, find which point index contributed the maximum (via `argmax`)
3. Collect all unique point indices → these are the **critical points**

### Visualizations (5 test samples)
Each visualization shows:
- **Left:** Full point cloud (1024 points, blue)
- **Right:** Critical points (red, highlighted over faint grey full cloud)

| Sample | Class | Critical Points (unique) |
|--------|-------|--------------------------|
| 1 | Sofa | 394 / 1024 |
| 2 | Bed | 437 / 1024 |
| 3 | Bathtub | 438 / 1024 |
| 4 | Table | 363 / 1024 |
| 5 | Desk | 362 / 1024 |

**Average critical points per sample (across all test set):** 398.0 / 1024 (~38.9%)

**Observation:** The critical points tend to concentrate at the geometrically distinctive parts of each object — edges, corners, and surfaces with high curvature — rather than being uniformly distributed. This suggests the network has learned to focus on the most informative structural features.

### Robustness Experiment (critical-points-only input)
- **Accuracy on full cloud:** 0.9020 (819/908)
- **Accuracy on sparse critical-points-only cloud:** 0.8987 (816/908)
- **Accuracy drop:** 0.33% (only 3 additional misclassifications)

**Does the accuracy drop? Why or why not?**
The accuracy drops by only 0.33%, which is negligible. This strongly validates the theoretical property of PointNet's max-pooling architecture:

- The **global feature vector** is formed by taking the max across all N points for each of the 1024 feature channels.
- By definition, only the points that achieve the maximum in at least one channel (the critical points) contribute to the global feature.
- All remaining non-critical points are effectively "**invisible**" to the model — they provide no information to the classifier.
- Therefore, using only the critical points (~398/1024 ≈ 39% of points) preserves nearly identical accuracy.

The minimal 0.33% drop is likely due to the slight distributional difference introduced by (1) re-centring and re-scaling the sparse cloud, and (2) padding the sparse cloud back to 1024 points via point repetition — both of which introduce minor numerical differences in the per-point features.
