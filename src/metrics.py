"""
metrics.py — Segmentation Evaluation Metrics & Compound Loss
============================================================
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
- Provide `SoftDiceLoss` and compound `BCEDiceLoss` for addressing extreme
  foreground/background class imbalance in ultrasound frames.

Public API
----------
    def dice_coefficient(pred, target, threshold=0.5, smooth=1e-6)
    def iou_score(pred, target, threshold=0.5, smooth=1e-6)
    class SoftDiceLoss(nn.Module)
    class BCEDiceLoss(nn.Module)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _prepare_inputs(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ensure prediction and target are binary float tensors with matching shapes.

    If pred contains values outside [0, 1] (raw logits), sigmoid is applied.
    Target is cast to float and thresholded at 0.5 if not already binary.
    """
    if pred.dtype != torch.float32 and pred.dtype != torch.float64:
        pred = pred.float()
    if target.dtype != torch.float32 and target.dtype != torch.float64:
        target = target.float()

    # Apply sigmoid if logits are detected
    if (pred < 0.0).any() or (pred > 1.0).any():
        pred_probs = torch.sigmoid(pred)
    else:
        pred_probs = pred

    pred_bin = (pred_probs > threshold).float()
    target_bin = (target > 0.5).float()

    # Match target dimension if pred is (B, 1, H, W) and target is (B, H, W)
    if pred_bin.ndim == target_bin.ndim + 1 and pred_bin.shape[1] == 1:
        target_bin = target_bin.unsqueeze(1)
    elif target_bin.ndim == pred_bin.ndim + 1 and target_bin.shape[1] == 1:
        pred_bin = pred_bin.unsqueeze(1)

    return pred_bin, target_bin


def dice_coefficient(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Compute the Dice Similarity Coefficient (DSC / F1-score) for binary segmentation.

    Dice = (2 * |pred ∩ target| + smooth) / (|pred| + |target| + smooth)

    Parameters
    ----------
    pred : torch.Tensor
        Predicted masks or raw logits. Shape: (B, C, H, W) or (B, H, W).
    target : torch.Tensor
        Ground truth binary masks. Shape: (B, C, H, W) or (B, H, W).
    threshold : float, default 0.5
        Binarization threshold for prediction probabilities.
    smooth : float, default 1e-6
        Smoothing epsilon to prevent division by zero.

    Returns
    -------
    torch.Tensor
        Scalar tensor containing the batch-averaged Dice coefficient.
    """
    pred_bin, target_bin = _prepare_inputs(pred, target, threshold=threshold)

    batch_size = pred_bin.shape[0]
    pred_flat = pred_bin.reshape(batch_size, -1)
    target_flat = target_bin.reshape(batch_size, -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    cardinality = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    return dice.mean()


def iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Compute the Intersection over Union (IoU / Jaccard Index) for binary segmentation.

    IoU = (|pred ∩ target| + smooth) / (|pred ∪ target| + smooth)
        = (|pred ∩ target| + smooth) / (|pred| + |target| - |pred ∩ target| + smooth)

    Parameters
    ----------
    pred : torch.Tensor
        Predicted masks or raw logits. Shape: (B, C, H, W) or (B, H, W).
    target : torch.Tensor
        Ground truth binary masks. Shape: (B, C, H, W) or (B, H, W).
    threshold : float, default 0.5
        Binarization threshold for prediction probabilities.
    smooth : float, default 1e-6
        Smoothing epsilon to prevent division by zero.

    Returns
    -------
    torch.Tensor
        Scalar tensor containing the batch-averaged IoU score.
    """
    pred_bin, target_bin = _prepare_inputs(pred, target, threshold=threshold)

    batch_size = pred_bin.shape[0]
    pred_flat = pred_bin.reshape(batch_size, -1)
    target_flat = target_bin.reshape(batch_size, -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    total = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    union = total - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()


class SoftDiceLoss(nn.Module):
    """Soft Dice Loss computed directly from continuous probabilities.

    Differentiable surrogate of the Dice Similarity Coefficient:
        L_Dice = 1 - (2 * sum(p * y) + smooth) / (sum(p) + sum(y) + smooth)

    Parameters
    ----------
    smooth : float, default 1e-6
        Smoothing constant to avoid division by zero.
    p_squared : bool, default False
        If True, uses squared terms in the denominator (V-Net variant).
    """

    def __init__(self, smooth: float = 1e-6, p_squared: bool = False):
        super().__init__()
        self.smooth = smooth
        self.p_squared = p_squared

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute soft dice loss.

        Parameters
        ----------
        logits : torch.Tensor
            Raw network predictions before sigmoid. Shape: (B, 1, H, W) or (B, H, W).
        target : torch.Tensor
            Ground truth binary mask in [0, 1]. Shape: (B, 1, H, W) or (B, H, W).

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        probs = torch.sigmoid(logits)

        if target.ndim == probs.ndim - 1:
            target = target.unsqueeze(1)
        target = target.float()

        batch_size = probs.shape[0]
        probs_flat = probs.reshape(batch_size, -1)
        target_flat = target.reshape(batch_size, -1)

        intersection = (probs_flat * target_flat).sum(dim=1)

        if self.p_squared:
            cardinality = (probs_flat ** 2).sum(dim=1) + (target_flat ** 2).sum(dim=1)
        else:
            cardinality = probs_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return (1.0 - dice).mean()


class BCEDiceLoss(nn.Module):
    """Compound loss combining BCEWithLogitsLoss and SoftDiceLoss.

    Addresses the severe foreground/background class imbalance in ultrasound
    scans by combining pixel-level smooth cross-entropy with global region overlap:
        L_total = alpha * L_BCE + beta * L_Dice

    Parameters
    ----------
    alpha : float, default 0.5
        Weight for BCEWithLogitsLoss.
    beta : float, default 0.5
        Weight for SoftDiceLoss.
    smooth : float, default 1e-6
        Smoothing epsilon for SoftDiceLoss.
    p_squared : bool, default False
        Whether to square sums in the soft dice denominator.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        smooth: float = 1e-6,
        p_squared: bool = False,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = SoftDiceLoss(smooth=smooth, p_squared=p_squared)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute compound BCE + Dice loss.

        Parameters
        ----------
        logits : torch.Tensor
            Raw output logits from the network (B, 1, H, W) or (B, H, W).
        target : torch.Tensor
            Binary ground truth masks (B, 1, H, W) or (B, H, W).

        Returns
        -------
        torch.Tensor
            Weighted compound loss scalar.
        """
        if target.ndim == logits.ndim - 1:
            target = target.unsqueeze(1)
        target = target.float()

        bce = self.bce_loss(logits, target)
        dice = self.dice_loss(logits, target)

        return self.alpha * bce + self.beta * dice
