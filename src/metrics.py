"""
metrics.py — Segmentation Evaluation Metrics
=============================================
Track 2B: Thyroid Nodule Ultrasound Segmentation (TN3k)

Responsibilities
----------------
- Compute the **Dice Coefficient** (F1-score for binary segmentation):
      Dice = 2 |A ∩ B| / (|A| + |B|)
- Compute **Intersection over Union** (Jaccard Index):
      IoU  = |A ∩ B| / |A ∪ B|
- Support both per-sample and batch-averaged computation.
- Accept raw logits or thresholded predictions; apply sigmoid + threshold
  internally when needed.
- Provide a small `smooth` term (default 1e-6) to avoid division by zero.

Expected public API
-------------------
    def dice_coefficient(pred, target, threshold=0.5, smooth=1e-6): ...
    def iou_score(pred, target, threshold=0.5, smooth=1e-6): ...
"""
