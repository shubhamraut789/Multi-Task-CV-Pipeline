# Q2 Report: Multi-Task Learning (Segmentation + Depth)

## Required WandB Links
- Vanilla Multi-Task U-Net: https://wandb.ai/sauravdeshmukh200-iiit-hyderabad/CV_A3_Q2/runs/4q9f7nmb
- U-Net without Skip Connections: https://wandb.ai/sauravdeshmukh200-iiit-hyderabad/CV_A3_Q2/runs/e8z8ziem
- U-Net with Residual Blocks: https://wandb.ai/sauravdeshmukh200-iiit-hyderabad/CV_A3_Q2/runs/mm46j3cx
- Combined Report: https://api.wandb.ai/links/sauravdeshmukh200-iiit-hyderabad/cwtyydvx

## Folder Map
- vanialla_unet/
- without_skip/
- residual/
- comparisons/

## What Each Image Represents

For each model folder (vanialla_unet, without_skip, residual):

1. losses.png
   - 4-panel figure showing train/validation curves for: total loss, segmentation CE loss, depth RMSE loss, and an overlay of individual losses.

2. mIOU_plot.png
   - Train and validation mIoU (Mean Intersection over Union) curves over epochs.

3. RMSE_plot.png
   - Train and validation depth RMSE curves over epochs (lower is better).

4. qualitative_results.png
   - Grid of 10 test-set samples, each showing side-by-side: Input Image | GT Seg Mask | Predicted Seg Mask | GT Depth | Predicted Depth.

## 2.1 Vanilla Multi-Task U-Net
- **Combined loss used (formula and coefficients):**
  `Total = 1.0 × CrossEntropy(seg) + 0.1 × RMSE(depth)`
- **Why this combination:**
  CrossEntropy is the standard loss for multi-class segmentation; RMSE is interpretable for depth regression. Depth weight 0.1 was chosen because the depth RMSE values are much smaller in magnitude than CE and the segmentation task dominates otherwise. This coefficient balances the two gradients effectively.
- **Training config:** batch_size=16, epochs=30, lr=1e-3, optimizer=Adam w/ weight_decay=1e-5, scheduler=ReduceLROnPlateau (patience=5)
- **Model parameters:** 31,385,743
- **Best val mIoU:** 0.7806 (epoch 27)
- **Final test metrics:**
    - mIoU: **0.7810**
    - RMSE: **0.0334**
    - Final test loss: **0.0641**

## 2.2 U-Net without Skip Connections

- **Training config:** batch_size=16, epochs=20, seg_weight=1.0, depth_weight=0.1 (same as vanilla)
- **Model parameters:** 28,252,303
- **Best val mIoU:** 0.5761 (epoch 20)
- **Final test metrics:**
    - mIoU: **0.5738**
    - RMSE: **0.0380**
    - Final test loss: **0.1851**
- **Comparison with vanilla (brief):**
    - **Segmentation boundaries:** Significantly worse — mIoU drops from 0.7810 → 0.5738 (−26.5%). Without skip connections, fine-grained spatial details from the encoder are lost, leading to blurry and misaligned segmentation boundaries, especially around thin objects (poles, signs) and irregular shapes (vegetation edges).
    - **Depth quality:** RMSE increases from 0.0334 → 0.0380 (13.8% worse). Depth predictions lose sharpness at object edges and depth discontinuities because high-resolution features are absent in the decoder.

## 2.3 U-Net with Residual Connections
- **Residual block summary:**
  Each double-conv block is replaced with a Residual Block: two 3×3 conv layers with BatchNorm + ReLU, plus a skip (shortcut) connection that adds the block input to its output. When input and output channel dimensions differ, a 1×1 convolution is used in the shortcut path. This enables deeper gradient flow and richer feature extraction.
- **Training config:** batch_size=16, epochs=25, seg_weight=1.0, depth_weight=0.1, lr=1e-3 → reduced to 5e-4 by ReduceLROnPlateau
- **Model parameters:** 33,132,623
- **Best val mIoU:** 0.8139 (epoch 25)
- **Final test metrics:**
    - mIoU: **0.8161**
    - RMSE: **0.0282**
    - Final test loss: **0.0493**
- **Comparison with vanilla (brief):**
    - **Segmentation boundaries:** Better — mIoU improves from 0.7810 → 0.8161 (+4.5%). Residual connections enable the network to learn more discriminative features, resulting in sharper class boundaries and better handling of small objects.
    - **Depth quality:** Noticeably better — RMSE decreases from 0.0334 → 0.0282 (15.6% improvement). Depth maps show crisper edges and more accurate distance estimation, especially at object boundaries and in far-field regions.
    - **Training observations:** A brief instability spike occurred around epoch 18 (likely due to gradient explosion on a difficult batch), but the ReduceLROnPlateau scheduler helped recovery. The model continued to improve through epoch 25.

## Comparison Table

| Model Variant       | mIoU   | RMSE   | Final Test Loss |
|---------------------|--------|--------|-----------------|
| Vanilla UNet        | 0.7810 | 0.0334 | 0.0641          |
| Without skip UNet   | 0.5738 | 0.0380 | 0.1851          |
| With Residual UNet  | 0.8161 | 0.0282 | 0.0493          |

### Analysis
1. **Skip connections are critical:** Removing them causes a 26.5% drop in mIoU and 13.8% increase in RMSE. This confirms that spatial feature propagation from encoder to decoder is essential for both tasks.
2. **Residual blocks outperform vanilla:** The residual architecture achieves the best results across all metrics. The shortcut connections within each block enable better gradient flow and richer feature learning, particularly beneficial for the depth task (15.6% RMSE improvement).
3. **Residual > Vanilla > No-skip:** This ordering is consistent across all metrics (mIoU, RMSE, loss), confirming that architectural depth and feature propagation strategies have significant impact on multi-task learning performance.

## Comparison Images (in comparisons/)
- comparison_plots_if_any.png
    - Bar chart showing test-set mIoU, RMSE, and loss side-by-side for all three architectures.
- comparison_results_if_any.png
    - Overlay of validation mIoU and RMSE curves across epochs for all three models, showing convergence behaviour and relative performance.
