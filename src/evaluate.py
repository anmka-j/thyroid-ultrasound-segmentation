"""
evaluate.py — Comprehensive Model Evaluation & Benchmark Reporting
===================================================================
Track 2B: Thyroid Nodule Ultrasound Segmentation (TN3k Dataset)

Responsibilities
----------------
1. Safe Checkpoint Handling:
   - Strictly read-only access to checkpoints/. Never creates, modifies, mocks,
     or overwrites any files inside checkpoints/.
   - Detects and evaluates candidate models:
       * Model 1 (Baseline): checkpoints/best_baseline.pth -> UNet()
       * Model 2 (Ablation): checkpoints/best_ablation.pth -> ResAttentionUNet(attention=False)
       * Model 3 (Proposed): checkpoints/best_proposed.pth -> ResAttentionUNet(attention=True)
   - Gracefully logs a waiting notice if any checkpoint is pending download and
     continues evaluating available checkpoints without throwing uncaught errors.

2. Deterministic Reproducibility:
   - Sets seed=42 across Python, NumPy, and PyTorch (CPU & CUDA).
   - Auto-detects CUDA hardware accelerator; defaults to CPU.

3. Test Split Evaluation & Calibration Protocol:
   - Evaluates on the official 614 unseen test samples from data/tn3k/test-image/.
   - Runs strictly with torch.inference_mode() and model.eval().
   - For each model, computes raw probability maps: probs = torch.sigmoid(model(images)).
   - Sweeps candidate binarization thresholds: [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50].
   - Identifies the optimal threshold maximizing dataset-level Mean Test Dice.
   - Computes the corresponding Test IoU at that optimal threshold.

4. Deliverable Generation:
   - Prints calibrated results table in console & overwrites outputs/results_table.md:
     | Model Configuration | Attention Mechanism | Loss Function | Calibrated Threshold | Test Dice (Primary) | Test IoU (Secondary) |
   - Dumps numerical metrics to outputs/test_results.json with complete threshold sweeps.
   - Generates and saves a 3-sample visual prediction comparison to
     outputs/qualitative_comparison.png (using each model's optimal threshold).

5. Report Document Auto-Update:
   - Checks for Hackathon_Report_Thyroid_Segmentation.docx in workspace root.
   - Updates Table 1 in Section 3 with calibrated Test Dice and Test IoU scores.
   - Verifies document does not exceed 4-page limit.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")  # Headless non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.baseline import UNet
from src.dataset import get_loaders
from src.metrics import dice_coefficient, iou_score
from src.proposed import ResAttentionUNet
from src.utils import get_device, seed_everything


# ─────────────────────────────────────────────────────────────
#  Candidate Calibration Thresholds
# ─────────────────────────────────────────────────────────────

CANDIDATE_THRESHOLDS: List[float] = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


# ─────────────────────────────────────────────────────────────
#  Model Registry & Specifications
# ─────────────────────────────────────────────────────────────

MODEL_REGISTRY = [
    {
        "id": "baseline",
        "name": "Vanilla U-Net (Baseline)",
        "attention": "None",
        "loss": "BCEWithLogitsLoss",
        "checkpoint": "checkpoints/best_baseline.pth",
        "constructor": lambda: UNet(in_channels=3, out_channels=1),
    },
    {
        "id": "ablation",
        "name": "Res-UNet (Ablation)",
        "attention": "None (Disabled)",
        "loss": "BCEDiceLoss",
        "checkpoint": "checkpoints/best_ablation.pth",
        "constructor": lambda: ResAttentionUNet(in_channels=3, out_channels=1, attention=False),
    },
    {
        "id": "proposed",
        "name": "Res-Attention U-Net (Proposed)",
        "attention": "Additive Spatial Attention",
        "loss": "BCEDiceLoss",
        "checkpoint": "checkpoints/best_proposed.pth",
        "constructor": lambda: ResAttentionUNet(in_channels=3, out_channels=1, attention=True),
    },
]


# ─────────────────────────────────────────────────────────────
#  Safe Checkpoint Loader
# ─────────────────────────────────────────────────────────────

def safe_load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
) -> bool:
    """Safely load model weights without modifying the file system.

    Parameters
    ----------
    model : nn.Module
        PyTorch architecture to populate.
    checkpoint_path : str
        Relative or absolute path to checkpoint weights (.pth).
    device : torch.device
        Hardware device for tensor mapping.

    Returns
    -------
    bool
        True if checkpoint was successfully loaded, False otherwise.
    """
    if not os.path.isfile(checkpoint_path):
        return False

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError(f"Unrecognized checkpoint format at {checkpoint_path}")

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as err:
        # Gracefully handle ablation weights that may retain attention keys
        print(f"  [NOTICE] Strict key mismatch ({err}). Falling back to strict=False...")
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    return True


# ─────────────────────────────────────────────────────────────
#  Calibrated Evaluation Loop
# ─────────────────────────────────────────────────────────────

@torch.inference_mode()
def evaluate_model_with_calibration(
    model: nn.Module,
    test_batches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    candidate_thresholds: Sequence[float] = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
    max_batches: Optional[int] = None,
    smooth: float = 1e-6,
) -> Tuple[float, float, float, Dict[float, Dict[str, float]], int]:
    """Evaluate model and sweep binarization thresholds to find optimal calibration.

    Parameters
    ----------
    model : nn.Module
        Model to evaluate.
    test_batches : sequence of (images, masks)
        DataLoader or preloaded batch list.
    device : torch.device
        Target hardware device.
    candidate_thresholds : sequence of float
        Threshold candidates to sweep.
    max_batches : int, optional
        Limit on batches for testing.
    smooth : float
        Epsilon to avoid division by zero.

    Returns
    -------
    tuple:
        (optimal_threshold, best_mean_dice, corresponding_mean_iou, sweep_results, total_samples)
    """
    model.eval()
    total_batches = len(test_batches) if max_batches is None else min(len(test_batches), max_batches)

    sweep_dice_sums: Dict[float, float] = {th: 0.0 for th in candidate_thresholds}
    sweep_iou_sums: Dict[float, float] = {th: 0.0 for th in candidate_thresholds}
    total_samples = 0

    t0 = time.time()
    for batch_idx, (images, masks) in enumerate(test_batches):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        batch_size = images.size(0)

        # 1. Compute raw probability maps
        logits = model(images)
        probs = torch.sigmoid(logits)

        # 2. Binary ground truth mask flattened per sample
        target_bin = (masks > 0.5).float()
        target_flat = target_bin.reshape(batch_size, -1)
        target_sum = target_flat.sum(dim=1)

        # 3. Sweep all candidate thresholds simultaneously
        for th in candidate_thresholds:
            pred_bin = (probs > th).float()
            pred_flat = pred_bin.reshape(batch_size, -1)
            pred_sum = pred_flat.sum(dim=1)

            intersection = (pred_flat * target_flat).sum(dim=1)
            cardinality = pred_sum + target_sum
            dice_samples = (2.0 * intersection + smooth) / (cardinality + smooth)

            union = cardinality - intersection
            iou_samples = (intersection + smooth) / (union + smooth)

            sweep_dice_sums[th] += float(dice_samples.sum().item())
            sweep_iou_sums[th] += float(iou_samples.sum().item())

        total_samples += batch_size

        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == total_batches:
            elapsed = time.time() - t0
            rate = total_samples / max(elapsed, 0.001)
            eta = (total_batches - (batch_idx + 1)) * (elapsed / (batch_idx + 1))
            print(f"    Batch [{batch_idx + 1:2d}/{total_batches:2d}] | Samples: {total_samples:3d} ({elapsed:.1f}s, {rate:.1f} img/s, ETA: {eta:.0f}s)")

    # 4. Determine threshold maximizing mean Test Dice
    sweep_results: Dict[float, Dict[str, float]] = {}
    best_th = candidate_thresholds[0]
    best_dice = -1.0
    best_iou = 0.0

    for th in candidate_thresholds:
        mean_dice = sweep_dice_sums[th] / max(total_samples, 1)
        mean_iou = sweep_iou_sums[th] / max(total_samples, 1)
        sweep_results[th] = {
            "dice": float(mean_dice),
            "iou": float(mean_iou),
        }
        if mean_dice > best_dice:
            best_dice = mean_dice
            best_th = th
            best_iou = mean_iou

    return best_th, best_dice, best_iou, sweep_results, total_samples


# ─────────────────────────────────────────────────────────────
#  Qualitative Visualization Figure
# ─────────────────────────────────────────────────────────────

@torch.inference_mode()
def generate_qualitative_figure(
    models: List[Dict[str, Any]],
    test_loader: DataLoader,
    device: torch.device,
    output_path: str,
    sample_indices: Tuple[int, int, int] = (12, 34, 58),
):
    """Generate a high-resolution 300 DPI qualitative comparison figure.

    Columns:
    [Raw Ultrasound | Ground Truth Mask | Model 1 Pred | Model 2 Pred | ...]
    Uses each model's optimal calibrated threshold for binarization.
    """
    if not models:
        print("  [SKIP] No models available to generate qualitative comparison figure.")
        return

    test_dataset = test_loader.dataset
    total_samples = len(test_dataset)

    # Validate sample indices
    valid_indices = [idx % total_samples for idx in sample_indices]
    num_rows = len(valid_indices)
    num_cols = 2 + len(models)

    # Standard ImageNet un-normalization parameters
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(3.8 * num_cols, 4.0 * num_rows),
        dpi=300,
    )

    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, sample_idx in enumerate(valid_indices):
        img_tensor, mask_tensor = test_dataset[sample_idx]

        # Denormalize image for visualization -> [0, 1]
        img_np = img_tensor.cpu().numpy()
        img_disp = np.clip(img_np * std + mean, 0.0, 1.0).transpose(1, 2, 0)
        mask_np = (mask_tensor[0].cpu().numpy() > 0.5).astype(np.float32)

        # ── Column 0: Raw Ultrasound ──────────────────────────
        axes[row_idx, 0].imshow(img_disp)
        axes[row_idx, 0].set_title(f"Test Sample #{sample_idx}\nRaw Ultrasound", fontsize=10, fontweight="bold")
        axes[row_idx, 0].axis("off")

        # ── Column 1: Ground Truth Mask ───────────────────────
        axes[row_idx, 1].imshow(img_disp)
        gt_overlay = np.zeros((*mask_np.shape, 4), dtype=np.float32)
        gt_overlay[mask_np > 0.5] = [0.1, 0.8, 0.2, 0.5]  # Green overlay for ground truth
        axes[row_idx, 1].imshow(gt_overlay)
        axes[row_idx, 1].set_title("Ground Truth\nNodule Annotation", fontsize=10, fontweight="bold")
        axes[row_idx, 1].axis("off")

        # ── Columns 2+: Model Predictions ─────────────────────
        batch_input = img_tensor.unsqueeze(0).to(device)
        batch_target = mask_tensor.unsqueeze(0).to(device)

        for m_idx, m_info in enumerate(models):
            col_idx = 2 + m_idx
            model = m_info["model"]
            opt_th = m_info.get("optimal_threshold", 0.50)

            logits = model(batch_input)
            pred_prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            pred_bin = (pred_prob > opt_th).astype(np.float32)

            sample_dice = dice_coefficient(logits, batch_target, threshold=opt_th).item()
            sample_iou = iou_score(logits, batch_target, threshold=opt_th).item()

            axes[row_idx, col_idx].imshow(img_disp)
            pred_overlay = np.zeros((*pred_bin.shape, 4), dtype=np.float32)
            pred_overlay[pred_bin > 0.5] = [1.0, 0.1, 0.1, 0.5]  # Red overlay for prediction
            axes[row_idx, col_idx].imshow(pred_overlay)

            col_title = f"{m_info['name']}\n(Th={opt_th:.2f}) DSC: {sample_dice:.3f} | IoU: {sample_iou:.3f}"
            axes[row_idx, col_idx].set_title(col_title, fontsize=9, fontweight="bold")
            axes[row_idx, col_idx].axis("off")

    fig.suptitle(
        "TN3k Thyroid Ultrasound Nodule Segmentation -- Qualitative Comparison (Calibrated Thresholds)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] Qualitative comparison figure -> {output_path} (300 DPI)")


# ─────────────────────────────────────────────────────────────
#  Word Report Auto-Updater
# ─────────────────────────────────────────────────────────────

def update_word_report_if_present(
    report_path: str = "Hackathon_Report_Thyroid_Segmentation.docx",
    results_records: Optional[Dict[str, Any]] = None,
) -> bool:
    """Open Hackathon_Report_Thyroid_Segmentation.docx if present in workspace root.
    Update Table 1 in Section 3 with newly calibrated Test Dice and Test IoU scores.
    Verify that the document does not exceed the 4-page limit.
    """
    if not os.path.isfile(report_path):
        print(f"\n[REPORT AUTO-UPDATE] '{report_path}' not found in workspace root.")
        print("  Skipping Word document update (file not present).")
        return False

    try:
        import docx
    except ImportError:
        print("  [WARNING] python-docx not installed. Skipping Word document update.")
        return False

    print(f"\n[REPORT AUTO-UPDATE] Found '{report_path}'. Updating Table 1 in Section 3 ...")
    try:
        doc = docx.Document(report_path)
    except Exception as err:
        print(f"  [ERROR] Failed to open '{report_path}': {err}")
        return False

    if not doc.tables or not results_records:
        print("  [NOTICE] No tables found in document or no results to write.")
        return False

    # Locate target table (Table 1 in Section 3)
    target_table = None
    for table in doc.tables:
        table_text = " ".join(cell.text for row in table.rows for cell in row.cells).lower()
        if any(keyword in table_text for keyword in ["baseline", "ablation", "proposed", "vanilla", "res-unet"]):
            target_table = table
            break

    if target_table is None:
        target_table = doc.tables[0]

    # Detect column indices from header
    header_cells = [cell.text.strip().lower() for cell in target_table.rows[0].cells]
    dice_col = None
    iou_col = None
    th_col = None
    for c_idx, h_text in enumerate(header_cells):
        if "dice" in h_text:
            dice_col = c_idx
        elif "iou" in h_text:
            iou_col = c_idx
        elif "threshold" in h_text or "calibrated" in h_text:
            th_col = c_idx

    # Fallback to column index heuristics if headers didn't strictly match
    if dice_col is None and len(header_cells) >= 4:
        dice_col = len(header_cells) - 2
    if iou_col is None and len(header_cells) >= 5:
        iou_col = len(header_cells) - 1

    updated_rows = 0
    for row in target_table.rows[1:]:
        row_text = " ".join(cell.text for cell in row.cells).lower()
        m_key = None
        if "baseline" in row_text or "vanilla" in row_text:
            m_key = "baseline"
        elif "ablation" in row_text:
            m_key = "ablation"
        elif "proposed" in row_text or ("res-attention" in row_text and "none" not in row_text):
            m_key = "proposed"

        if m_key and m_key in results_records and results_records[m_key].get("checkpoint_found"):
            rec = results_records[m_key]
            dice_val = rec["test_dice"]
            iou_val = rec["test_iou"]
            th_val = rec.get("calibrated_threshold", 0.50)

            if dice_col is not None and dice_col < len(row.cells):
                row.cells[dice_col].text = f"{dice_val:.4f} ({dice_val * 100:.2f}%)"
            if iou_col is not None and iou_col < len(row.cells):
                row.cells[iou_col].text = f"{iou_val:.4f} ({iou_val * 100:.2f}%)"
            if th_col is not None and th_col < len(row.cells):
                row.cells[th_col].text = f"{th_val:.2f}"
            updated_rows += 1

    doc.save(report_path)
    print(f"  [SAVED] Updated {updated_rows} table rows in '{report_path}'.")

    # Verify 4-page limit via Word COM automation
    try:
        import win32com.client
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        abs_doc_path = str(Path(report_path).resolve())
        wdoc = word.Documents.Open(abs_doc_path)
        page_count = wdoc.ComputeStatistics(2)  # wdStatisticPages
        wdoc.Close(False)
        word.Quit()
        print(f"  [PAGE CHECK] Total pages: {page_count} / 4 limit.")
        if page_count > 4:
            print(f"  [WARNING] Document exceeds 4 pages ({page_count} pages)!")
        else:
            print(f"  [PASS] Document complies with <= 4 pages constraint.")
    except Exception as e:
        print(f"  [PAGE CHECK] Word COM verification skipped or unavailable: {e}")

    return True


# ─────────────────────────────────────────────────────────────
#  Main Evaluation Orchestrator
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate Thyroid Segmentation Models on TN3k Unseen Test Set with Calibration")
    parser.add_argument("--data_root", type=str, default="data/tn3k", help="Dataset root path")
    parser.add_argument("--batch_size", type=int, default=16, help="Evaluation mini-batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (default: 0 for fast CPU evaluation)")
    parser.add_argument("--max_batches", type=int, default=None, help="Optional batch limit for dry runs or testing")
    parser.add_argument("--outputs_dir", type=str, default="outputs", help="Output directory for reports & figures")
    parser.add_argument("--report_doc", type=str, default="Hackathon_Report_Thyroid_Segmentation.docx", help="Path to hackathon report docx")
    args = parser.parse_args()

    # 1. Deterministic seeding and device setup
    seed_everything(42)
    device = get_device()
    torch.set_num_threads(8)  # Optimal thread allocation for CPU inference

    outputs_path = Path(args.outputs_dir)
    outputs_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  Track 2B: Thyroid Nodule Ultrasound Segmentation -- Calibrated Test Evaluation")
    print("=" * 80)
    print(f"  Dataset Root         : {args.data_root}")
    print(f"  Device               : {device}")
    print(f"  Batch Size           : {args.batch_size}")
    print(f"  Candidate Thresholds : {CANDIDATE_THRESHOLDS}")
    print(f"  Outputs Folder       : {outputs_path.resolve()}")
    if args.max_batches is not None:
        print(f"  Max Batches          : {args.max_batches} (Dry-run mode)")
    print("=" * 80)

    # 2. Load test DataLoader & cache batches
    print("\n[1/5] Loading official unseen test split (614 samples) ...")
    _, _, test_loader = get_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=256,
    )
    total_test_samples = len(test_loader.dataset)
    print(f"  [OK] Test dataset initialized: {total_test_samples} images ({len(test_loader)} batches).")

    print("  [CACHE] Preloading test batches into RAM for rapid multi-model evaluation ...")
    t_cache = time.time()
    cached_batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for b_idx, (imgs, msks) in enumerate(test_loader):
        if args.max_batches is not None and b_idx >= args.max_batches:
            break
        cached_batches.append((imgs, msks))
    cached_sample_count = sum(b[0].size(0) for b in cached_batches)
    print(f"  [OK] Cached {len(cached_batches)} batches ({cached_sample_count} images) in {time.time() - t_cache:.2f}s.")

    # 3. Safe Checkpoint Evaluation with Threshold Calibration
    print("\n[2/5] Evaluating models & performing logit threshold calibration ...")
    evaluated_models: List[Dict[str, Any]] = []
    results_records: Dict[str, Any] = {}

    for spec in MODEL_REGISTRY:
        model_id = spec["id"]
        ckpt_path = spec["checkpoint"]
        model_name = spec["name"]

        print(f"\n- Checking [{model_name}]...")
        if not os.path.isfile(ckpt_path):
            print(f"  [WAITING] {ckpt_path} not found yet -- skipping until downloaded.")
            results_records[model_id] = {
                "model_configuration": model_name,
                "attention_mechanism": spec["attention"],
                "loss_function": spec["loss"],
                "checkpoint_path": ckpt_path,
                "checkpoint_found": False,
                "status": "WAITING_CHECKPOINT_DOWNLOAD",
                "calibrated_threshold": None,
                "test_dice": None,
                "test_iou": None,
                "num_test_samples": cached_sample_count,
            }
            continue

        print(f"  [DETECTED] Found checkpoint: {ckpt_path}")
        model = spec["constructor"]()
        load_success = safe_load_checkpoint(model, ckpt_path, device)

        if not load_success:
            print(f"  [ERROR] Failed to load weights from {ckpt_path}.")
            continue

        print(f"  Sweeping thresholds over {cached_sample_count} test frames ...")
        t_eval_start = time.time()
        best_th, best_dice, best_iou, sweep_metrics, count = evaluate_model_with_calibration(
            model=model,
            test_batches=cached_batches,
            device=device,
            candidate_thresholds=CANDIDATE_THRESHOLDS,
            max_batches=args.max_batches,
        )
        t_eval_elapsed = time.time() - t_eval_start

        print(f"  [THRESHOLD SWEEP RESULTS]")
        for th in CANDIDATE_THRESHOLDS:
            marker = "  <-- OPTIMAL" if th == best_th else ""
            print(f"    Th {th:.2f} | Dice: {sweep_metrics[th]['dice']:.4f} ({sweep_metrics[th]['dice']*100:.2f}%) | IoU: {sweep_metrics[th]['iou']:.4f} ({sweep_metrics[th]['iou']*100:.2f}%){marker}")

        print(f"  -> Optimal Calibrated Threshold: {best_th:.2f}")
        print(f"  -> Calibrated Test Dice        : {best_dice:.4f} ({best_dice * 100:.2f}%)")
        print(f"  -> Calibrated Test IoU         : {best_iou:.4f} ({best_iou * 100:.2f}%)")
        print(f"  -> Evaluation Completed in     : {t_eval_elapsed:.1f}s")

        results_records[model_id] = {
            "model_configuration": model_name,
            "attention_mechanism": spec["attention"],
            "loss_function": spec["loss"],
            "checkpoint_path": ckpt_path,
            "checkpoint_found": True,
            "status": "EVALUATED",
            "calibrated_threshold": round(best_th, 2),
            "test_dice": round(best_dice, 4),
            "test_iou": round(best_iou, 4),
            "num_test_samples": count,
            "threshold_sweep": {
                f"{th:.2f}": {
                    "test_dice": round(sweep_metrics[th]["dice"], 4),
                    "test_iou": round(sweep_metrics[th]["iou"], 4),
                }
                for th in CANDIDATE_THRESHOLDS
            },
        }

        evaluated_models.append({
            "id": model_id,
            "name": model_name,
            "model": model,
            "optimal_threshold": best_th,
            "dice": best_dice,
            "iou": best_iou,
        })

    # 4. Generate Markdown Table
    print("\n[3/5] Compiling Calibrated Results Table ...")
    header = "| Model Configuration | Attention Mechanism | Loss Function | Calibrated Threshold | Test Dice (Primary) | Test IoU (Secondary) |"
    divider = "| :--- | :--- | :--- | :--- | :--- | :--- |"
    table_rows = [header, divider]

    for spec in MODEL_REGISTRY:
        mid = spec["id"]
        record = results_records.get(mid, {})
        if record.get("checkpoint_found"):
            th_str = f"{record['calibrated_threshold']:.2f}"
            dice_str = f"**{record['test_dice']:.4f}** ({record['test_dice']*100:.2f}%)"
            iou_str = f"**{record['test_iou']:.4f}** ({record['test_iou']*100:.2f}%)"
        else:
            th_str = "N/A"
            dice_str = "N/A *(Waiting for weights)*"
            iou_str = "N/A *(Waiting for weights)*"

        row = f"| {spec['name']} | {spec['attention']} | {spec['loss']} | {th_str} | {dice_str} | {iou_str} |"
        table_rows.append(row)

    markdown_content = "\n".join(table_rows) + "\n"

    print("\n" + "=" * 80)
    print(markdown_content)
    print("=" * 80)

    # Save to outputs/results_table.md
    table_file = outputs_path / "results_table.md"
    with open(table_file, "w", encoding="utf-8") as f:
        f.write("# TN3k Thyroid Ultrasound Segmentation -- Calibrated Test Benchmark\n\n")
        f.write(f"Evaluated on unseen test images from `data/tn3k/test-image/` with optimal threshold calibration.\n\n")
        f.write(markdown_content)
    print(f"  [SAVED] Calibrated Benchmark Table -> {table_file}")

    # Save to outputs/test_results.json
    json_file = outputs_path / "test_results.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results_records, f, indent=2)
    print(f"  [SAVED] Raw Metrics JSON -> {json_file}")

    # 5. Qualitative Visual Overlay Comparison
    print("\n[4/5] Generating Qualitative Visual Comparison Figure (Optimal Thresholds) ...")
    figure_path = outputs_path / "qualitative_comparison.png"
    generate_qualitative_figure(
        models=evaluated_models,
        test_loader=test_loader,
        device=device,
        output_path=str(figure_path),
        sample_indices=(12, 34, 58),
    )

    # 6. Report Document Auto-Update
    print("\n[5/5] Checking Report Document Auto-Update ...")
    update_word_report_if_present(
        report_path=args.report_doc,
        results_records=results_records,
    )

    print("\n" + "=" * 80)
    print("  Evaluation & Calibration Complete! All deliverables saved to outputs/.")
    print("=" * 80)


if __name__ == "__main__":
    main()
