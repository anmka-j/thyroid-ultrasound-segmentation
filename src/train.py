"""
train.py — Central Model Training Engine
========================================
Track 2B: Thyroid Nodule Ultrasound Segmentation (TN3k)

Responsibilities
----------------
- Parse CLI arguments (--model, --epochs, --batch_size, --lr, --data_root, --dry_run).
- Enforce full reproducibility with seed locking (seed=42).
- Build and configure training/validation DataLoaders.
- Support ablation study mode (disables augmentations for train set).
- Model selection:
    * 'baseline': Vanilla U-Net with BCEWithLogitsLoss.
    * 'proposed' / 'ablation': ResAttentionUNet with BCEDiceLoss (or BCE fallback).
- Optimizer (AdamW, weight_decay=1e-4) and LR scheduler (ReduceLROnPlateau, mode='max', patience=3).
- Evaluation tracking (Train Loss, Val Loss, Val Dice).
- Model checkpointing (checkpoints/best_{model}.pth) and Early Stopping (patience=7).
- Dry run verification mode for rapid pipeline sanity testing.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.dataset import get_loaders, get_val_transform
from src.utils import get_device, seed_everything


# ─────────────────────────────────────────────────────────────
#  Metrics & Loss Helper
# ─────────────────────────────────────────────────────────────

def compute_dice_score(
    pred_logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> float:
    """Compute mean Dice Coefficient over a batch.

    Parameters
    ----------
    pred_logits : torch.Tensor
        Raw unactivated network predictions of shape (B, 1, H, W).
    targets : torch.Tensor
        Ground truth binary masks of shape (B, 1, H, W) with values in {0.0, 1.0}.
    threshold : float, default=0.5
        Sigmoid probability cutoff.
    smooth : float, default=1e-6
        Smoothing factor to prevent division by zero.

    Returns
    -------
    float
        Batch-averaged Dice score.
    """
    probs = torch.sigmoid(pred_logits)
    preds = (probs > threshold).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    total = preds.sum(dim=1) + targets.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (total + smooth)
    return float(dice.mean().item())


class FallbackBCEDiceLoss(nn.Module):
    """Combined BCE and Soft Dice Loss fallback if src.metrics is not yet merged."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        cardinality = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss.mean()


# ─────────────────────────────────────────────────────────────
#  Model & Criterion Factory
# ─────────────────────────────────────────────────────────────

def get_model_and_criterion(
    model_name: str,
    device: torch.device,
) -> Tuple[nn.Module, nn.Module]:
    """Instantiate model and loss function based on configuration.

    Parameters
    ----------
    model_name : str
        One of 'baseline', 'proposed', or 'ablation'.
    device : torch.device
        Target hardware device.

    Returns
    -------
    tuple[nn.Module, nn.Module]
        (model, criterion)
    """
    if model_name == "baseline":
        from src.baseline import UNet
        model = UNet(in_channels=3, out_channels=1).to(device)
        criterion = nn.BCEWithLogitsLoss()
        print(f"[INFO] Loaded Model: Vanilla UNet (Baseline) with BCEWithLogitsLoss")

    elif model_name in ["proposed", "ablation"]:
        try:
            from src.proposed import ResAttentionUNet
            model = ResAttentionUNet(in_channels=3, out_channels=1).to(device)
        except (ImportError, AttributeError):
            print("[WARN] ResAttentionUNet not yet implemented in src.proposed. Falling back to UNet for testing.")
            from src.baseline import UNet
            model = UNet(in_channels=3, out_channels=1).to(device)

        try:
            from src.metrics import BCEDiceLoss
            criterion = BCEDiceLoss()
            print(f"[INFO] Loaded BCEDiceLoss from src.metrics")
        except (ImportError, AttributeError):
            print(f"[WARN] BCEDiceLoss not found in src.metrics. Using FallbackBCEDiceLoss.")
            criterion = FallbackBCEDiceLoss()

        print(f"[INFO] Loaded Model: {model.__class__.__name__} ({model_name})")

    else:
        raise ValueError(f"Unknown model name: {model_name}. Must be 'baseline', 'proposed', or 'ablation'.")

    return model, criterion


# ─────────────────────────────────────────────────────────────
#  Epoch Runners
# ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    dry_run: bool = False,
) -> float:
    """Train model for one epoch or execute a single-batch dry run."""
    model.train()
    running_loss = 0.0
    num_batches = 0

    for i, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

        if dry_run:
            print(f"  [DRY RUN] Train Batch 1/1: Loss = {loss.item():.4f}")
            return loss.item()

    return running_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    dry_run: bool = False,
) -> Tuple[float, float]:
    """Evaluate model on validation loader returning (val_loss, val_dice)."""
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    num_batches = 0

    for i, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)
        dice = compute_dice_score(logits, masks)

        running_loss += loss.item()
        running_dice += dice
        num_batches += 1

        if dry_run:
            print(f"  [DRY RUN] Val Batch 1/1: Loss = {loss.item():.4f}, Dice = {dice:.4f}")
            return loss.item(), dice

    avg_loss = running_loss / max(num_batches, 1)
    avg_dice = running_dice / max(num_batches, 1)
    return avg_loss, avg_dice


# ─────────────────────────────────────────────────────────────
#  Main Training Routine
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Central Training Engine for Thyroid Ultrasound Segmentation")
    parser.add_argument(
        "--model",
        type=str,
        default="baseline",
        choices=["baseline", "proposed", "ablation"],
        help="Model architecture / experiment variant (default: baseline)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=35,
        help="Number of training epochs (default: 35)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Mini-batch size (default: 16)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for AdamW optimizer (default: 1e-4)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/tn3k",
        help="Path to TN3k dataset root directory (default: data/tn3k)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of DataLoader worker processes (default: 2)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Execute 1 train batch and 1 val batch only, then exit",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Enforce reproducibility
    seed_everything(42)
    device = get_device()

    print("=" * 65)
    print("  Thyroid Ultrasound Segmentation -- Central Training Engine")
    print("=" * 65)
    print(f"  Model Variant  : {args.model}")
    print(f"  Total Epochs   : {args.epochs}")
    print(f"  Batch Size     : {args.batch_size}")
    print(f"  Learning Rate  : {args.lr}")
    print(f"  Data Root      : {args.data_root}")
    print(f"  Hardware Device: {device}")
    print(f"  Dry Run Mode   : {args.dry_run}")
    print("=" * 65)

    # 2. Prepare DataLoaders
    print("\n[1/4] Preparing DataLoaders ...")
    train_loader, val_loader, test_loader = get_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=256,
    )

    # Ablation Mode: disable training data augmentation
    if args.model == "ablation":
        print("[INFO] Ablation Mode Active: Overriding train loader to use val_transform (no augmentations)")
        val_tf = get_val_transform(img_size=256)
        train_loader.dataset.transform = val_tf

    print(f"  Train set: {len(train_loader.dataset)} samples ({len(train_loader)} batches)")
    print(f"  Val   set: {len(val_loader.dataset)} samples ({len(val_loader)} batches)")
    print(f"  Test  set: {len(test_loader.dataset)} samples ({len(test_loader)} batches)")

    # 3. Model & Criterion Selection
    print("\n[2/4] Initializing Model & Loss Criterion ...")
    model, criterion = get_model_and_criterion(args.model, device)

    # 4. Optimizer & Scheduler
    print("\n[3/4] Initializing Optimizer & LR Scheduler ...")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    print(f"  Optimizer: AdamW (lr={args.lr}, weight_decay=1e-4)")
    print(f"  Scheduler: ReduceLROnPlateau (mode=max, factor=0.5, patience=3)")

    # 5. Checkpointing setup
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / f"best_{args.model}.pth"

    # 6. Dry Run Execution
    if args.dry_run:
        print("\n[4/4] Executing Dry Run (1 train batch + 1 val batch) ...")
        t_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, dry_run=True)
        v_loss, v_dice = evaluate(model, val_loader, criterion, device, dry_run=True)
        print("\n=== DRY RUN COMPLETED SUCCESSFULLY ===")
        print(f"  Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | Val Dice: {v_dice:.4f}")
        print("Pipeline is verified and fully functional. Exiting cleanly.")
        return

    # 7. Full Training Loop
    print("\n[4/4] Commencing Model Training ...")
    best_val_dice = 0.0
    early_stop_patience = 7
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice = evaluate(model, val_loader, criterion, device)

        # Step scheduler based on validation Dice score
        scheduler.step(val_dice)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"Time: {elapsed:.1f}s",
            end="",
        )

        # Checkpoint if new best Val Dice
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": val_dice,
                    "val_loss": val_loss,
                    "model_name": args.model,
                },
                best_checkpoint_path,
            )
            print(f"  -> [SAVED best model (Dice: {val_dice:.4f})]")
        else:
            patience_counter += 1
            print(f"  (patience {patience_counter}/{early_stop_patience})")

        # Early Stopping
        if patience_counter >= early_stop_patience:
            print(f"\n[INFO] Early stopping triggered after {epoch} epochs. No improvement for {early_stop_patience} consecutive epochs.")
            break

    print("\n" + "=" * 65)
    print(f"  Training Finished!")
    print(f"  Best Validation Dice Score: {best_val_dice:.4f}")
    print(f"  Best Checkpoint Saved To  : {best_checkpoint_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
