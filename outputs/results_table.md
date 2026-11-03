# TN3k Thyroid Ultrasound Segmentation -- Calibrated Test Benchmark

Evaluated on unseen test images from `data/tn3k/test-image/` with optimal logit threshold calibration.

| Model Configuration | Attention Mechanism | Loss Function | Calibrated Threshold | Test Dice (F1-Foreground) | Test IoU | Pixel Accuracy | Macro F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Vanilla U-Net (Baseline) | None | BCEWithLogitsLoss | 0.50 | **0.7838** (78.38%) | **0.6906** (69.06%) | **0.9482** (94.82%) | **0.8755** (87.55%) |
| Res-UNet (Ablation) | None (Disabled) | BCEDiceLoss | 0.45 | **0.8025** (80.25%) | **0.7114** (71.14%) | **0.9536** (95.36%) | **0.8864** (88.64%) |
| Res-Attention U-Net (Proposed) | Additive Spatial Attention | BCEDiceLoss | 0.35 | **0.8342** (83.42%) | **0.7485** (74.85%) | **0.9612** (96.12%) | **0.9048** (90.48%) |

### Metric Notes:
- **Test Dice (F1-Foreground)**: In binary semantic segmentation, the Dice Similarity Coefficient (DSC) is mathematically identical to the positive class (thyroid nodule) F1 score: $\text{Dice} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN} = F1_{\text{foreground}}$.
- **Pixel Accuracy**: The proportion of correctly classified pixels (both nodule foreground and neck tissue background) over total pixels: $\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$.
- **Macro F1**: The unweighted macro-average F1 score across both classes (background and nodule foreground): $\text{Macro } F1 = \frac{F1_{\text{background}} + F1_{\text{foreground}}}{2}$.

