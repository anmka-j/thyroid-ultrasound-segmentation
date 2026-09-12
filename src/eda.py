"""
eda.py — Reusable Mathematical & Statistical Utilities for TN3k EDA
===================================================================
Track 2B: Thyroid Nodule Ultrasound Segmentation (TN3k)

Provides modular functions for pixel imbalance calculations and nodule
geometry / bounding box distribution analysis. Visualisation logic is
deferred to Jupyter notebooks.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image


def calculate_pixel_imbalance(mask_dir: str) -> Dict[str, Any]:
    """Iterate through all binary masks in a directory and calculate pixel imbalance.

    Parameters
    ----------
    mask_dir : str
        Directory containing binary mask images (.jpg / .png).

    Returns
    -------
    dict
        Dictionary containing:
        - "total_foreground": Total foreground (nodule) pixel count across the dataset.
        - "total_background": Total background pixel count across the dataset.
        - "total_pixels": Total pixels across all masks.
        - "global_fg_ratio": Global foreground-to-total ratio.
        - "global_bg_ratio": Global background-to-total ratio.
        - "per_mask_ratios": 1D numpy array of foreground ratios for each mask.
        - "mean_fg_ratio": Mean foreground ratio across masks.
        - "median_fg_ratio": Median foreground ratio across masks.
        - "min_fg_ratio": Minimum foreground ratio found in a mask.
        - "max_fg_ratio": Maximum foreground ratio found in a mask.
        - "std_fg_ratio": Standard deviation of foreground ratios.
        - "mask_names": List of mask filenames in sorted order.
    """
    valid_exts = {".jpg", ".jpeg", ".png"}
    mask_names = sorted([f for f in os.listdir(mask_dir) if Path(f).suffix.lower() in valid_exts])

    total_fg = 0
    total_bg = 0
    per_mask_ratios = []

    for fn in mask_names:
        mask_path = os.path.join(mask_dir, fn)
        # Read in grayscale
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.array(Image.open(mask_path).convert("L"))

        fg_count = int(np.sum(mask > 127))
        bg_count = int(mask.size - fg_count)

        total_fg += fg_count
        total_bg += bg_count
        per_mask_ratios.append(fg_count / mask.size)

    per_mask_ratios_arr = np.array(per_mask_ratios, dtype=np.float64)
    total_pixels = total_fg + total_bg

    return {
        "total_foreground": total_fg,
        "total_background": total_bg,
        "total_pixels": total_pixels,
        "global_fg_ratio": float(total_fg / total_pixels) if total_pixels > 0 else 0.0,
        "global_bg_ratio": float(total_bg / total_pixels) if total_pixels > 0 else 0.0,
        "per_mask_ratios": per_mask_ratios_arr,
        "mean_fg_ratio": float(np.mean(per_mask_ratios_arr)),
        "median_fg_ratio": float(np.median(per_mask_ratios_arr)),
        "min_fg_ratio": float(np.min(per_mask_ratios_arr)),
        "max_fg_ratio": float(np.max(per_mask_ratios_arr)),
        "std_fg_ratio": float(np.std(per_mask_ratios_arr)),
        "mask_names": mask_names,
    }


def get_nodule_size_distribution(
    mask_dir: str
) -> Dict[str, Any]:
    """Calculate nodule pixel areas and bounding box sizes across all masks.

    For each mask, connected components / contours are detected to identify
    nodule instances. Bounding box sizes are extracted as (width, height).

    Parameters
    ----------
    mask_dir : str
        Directory containing binary mask images (.jpg / .png).

    Returns
    -------
    dict
        Dictionary containing:
        - "areas": List of pixel counts for each nodule instance.
        - "bbox_sizes": List of (width, height) tuples for each bounding box.
        - "bbox_widths": List of bounding box widths.
        - "bbox_heights": List of bounding box heights.
        - "aspect_ratios": List of width/height aspect ratios.
        - "nodule_counts_per_mask": List of nodule count per mask image.
    """
    valid_exts = {".jpg", ".jpeg", ".png"}
    mask_names = sorted([f for f in os.listdir(mask_dir) if Path(f).suffix.lower() in valid_exts])

    areas: List[int] = []
    bbox_sizes: List[Tuple[int, int]] = []
    bbox_widths: List[int] = []
    bbox_heights: List[int] = []
    aspect_ratios: List[float] = []
    nodule_counts: List[int] = []

    for fn in mask_names:
        mask_path = os.path.join(mask_dir, fn)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.array(Image.open(mask_path).convert("L"))

        bin_mask = (mask > 127).astype(np.uint8)
        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        count = 0
        for cnt in contours:
            area = int(cv2.contourArea(cnt))
            if area <= 0:
                # Count non-zero if contourArea is 0 for tiny 1-2 pixel regions
                area = int(np.sum(bin_mask))
            if area > 0:
                x, y, w, h = cv2.boundingRect(cnt)
                areas.append(area)
                bbox_sizes.append((w, h))
                bbox_widths.append(w)
                bbox_heights.append(h)
                aspect_ratios.append(float(w / h) if h > 0 else 1.0)
                count += 1

        nodule_counts.append(count)

    return {
        "areas": areas,
        "bbox_sizes": bbox_sizes,
        "bbox_widths": bbox_widths,
        "bbox_heights": bbox_heights,
        "aspect_ratios": aspect_ratios,
        "nodule_counts_per_mask": nodule_counts,
    }
