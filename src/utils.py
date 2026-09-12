"""
utils.py — Reproducibility & Environment Utilities
====================================================
Track 2B: Thyroid Nodule Ultrasound Segmentation (TN3k)

Provides:
    • seed_everything(seed)  – locks all RNG sources for full reproducibility.
    • get_device()            – returns the best available PyTorch device.
"""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Lock every source of randomness for reproducible experiments.

    Sets seeds for Python's built-in `random`, NumPy, and all PyTorch backends
    (CPU + every visible CUDA device).  Also forces cuDNN into deterministic
    mode and disables its auto-tuner.

    Parameters
    ----------
    seed : int, default 42
        The global seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return the best available PyTorch compute device.

    Returns
    -------
    torch.device
        ``cuda`` when a CUDA-capable GPU is visible, otherwise ``cpu``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    return device
