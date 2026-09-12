"""
dataset.py — TN3k Dataset Loader
=================================
Track 2B: Thyroid Nodule Ultrasound Segmentation (TN3k)

Responsibilities
----------------
- Load thyroid-nodule ultrasound images and their corresponding binary masks
  from the TN3k dataset directory structure.
- Parse the official fold-0 JSON for an 80/20 train/val split (2,303 / 576).
- Resize all images and masks to 256x256 pixels.
- Convert single-channel grayscale to 3-channel RGB (replicate) for ImageNet
  pretrained backbone compatibility.
- Apply ImageNet normalisation to images.
- Strictly binarise masks to {0.0, 1.0}.
- Keep test-image / test-mask completely isolated from training and validation.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ─────────────────────────────────────────────────────────────
#  Transforms
# ─────────────────────────────────────────────────────────────

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def get_train_transform(img_size: int = 256) -> A.Compose:
    """Albumentations pipeline for training (geometric & photometric augmentations)."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transform(img_size: int = 256) -> A.Compose:
    """Albumentations pipeline for validation (deterministic resize & normalize)."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_test_transform(img_size: int = 256) -> A.Compose:
    """Albumentations pipeline for test evaluation (deterministic resize & normalize)."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────────

class TN3kDataset(Dataset):
    """PyTorch Dataset for the TN3k thyroid-nodule ultrasound dataset.

    Parameters
    ----------
    image_dir : str
        Path to the directory containing images (.jpg).
    mask_dir : str
        Path to the directory containing binary masks (.jpg).
    filenames_or_stems : list[str] or None
        Explicit list of filenames or stems to include (e.g. from a fold JSON).
        If *None*, all matching pairs in ``image_dir`` and ``mask_dir`` are used.
    transform : albumentations.Compose or None
        Augmentation / preprocessing pipeline applied jointly to image
        and mask.
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        filenames_or_stems: Optional[List[str]] = None,
        transform: Optional[A.Compose] = None,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # Index available images and masks by stem
        img_files = {Path(f).stem: f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
        mask_files = {Path(f).stem: f for f in os.listdir(mask_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}

        if filenames_or_stems is not None:
            # Map provided filenames or stems to matching stems
            self.stems = [Path(item).stem for item in filenames_or_stems]
        else:
            # Common stems between images and masks
            self.stems = sorted(list(set(img_files.keys()) & set(mask_files.keys())))

        self.img_map = img_files
        self.mask_map = mask_files

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        stem = self.stems[idx]
        img_fname = self.img_map.get(stem, f"{stem}.jpg")
        mask_fname = self.mask_map.get(stem, f"{stem}.jpg")

        img_path = os.path.join(self.image_dir, img_fname)
        mask_path = os.path.join(self.mask_dir, mask_fname)

        # 1. Read grayscale image and replicate to 3 channels (RGB)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # (H, W, 3), uint8

        # 2. Read mask as grayscale
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Failed to load mask: {mask_path}")

        # 3. Strictly binarise mask: > 127 -> 1.0, else 0.0
        mask = (mask > 127).astype(np.float32)  # (H, W), float32 strictly in {0.0, 1.0}

        # 4. Apply transforms (joint resize & normalize)
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]    # (3, 256, 256) float32
            mask = augmented["mask"]    # (256, 256) float32
        else:
            # Fallback resizing and ImageNet normalization
            img = cv2.resize(img, (256, 256))
            mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
            img = img.astype(np.float32) / 255.0
            img = (img - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).float()

        # Strictly re-verify binarization after any potential interpolation
        mask = (mask > 0.5).float()

        # Ensure channel dimension: (1, H, W)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return img, mask


# ─────────────────────────────────────────────────────────────
#  Loader Factory
# ─────────────────────────────────────────────────────────────

def get_loaders(
    data_root: str = "data/tn3k",
    batch_size: int = 16,
    num_workers: int = 2,
    img_size: int = 256,
    fold_json: str = "tn3k-trainval-fold0.json",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, validation, and test DataLoaders for TN3k.

    Uses the official fold JSON to determine the 80/20 trainval split
    (2,303 train / 576 val for fold-0).  The test split (614 images)
    is kept completely isolated.

    Parameters
    ----------
    data_root : str
        Root directory of the TN3k dataset.
    batch_size : int
        Mini-batch size.
    num_workers : int
        DataLoader worker processes.
    img_size : int
        Target spatial size (square, 256x256).
    fold_json : str
        Filename of the fold JSON inside ``data_root``.

    Returns
    -------
    tuple[DataLoader, DataLoader, DataLoader]
        ``(train_loader, val_loader, test_loader)``
    """
    # ── Parse official fold JSON ──────────────────────────────
    fold_path = os.path.join(data_root, fold_json)
    with open(fold_path, "r") as f:
        fold = json.load(f)

    # Sorted list of all 2,879 trainval filenames
    all_trainval_files = sorted(
        [fn for fn in os.listdir(os.path.join(data_root, "trainval-image"))
         if fn.lower().endswith(".jpg")]
    )
    train_stems = [all_trainval_files[i] for i in fold["train"]]
    val_stems   = [all_trainval_files[i] for i in fold["val"]]

    # ── Directories ───────────────────────────────────────────
    trainval_img_dir  = os.path.join(data_root, "trainval-image")
    trainval_mask_dir = os.path.join(data_root, "trainval-mask")
    test_img_dir      = os.path.join(data_root, "test-image")
    test_mask_dir     = os.path.join(data_root, "test-mask")

    # ── Transforms ────────────────────────────────────────────
    train_tf = get_train_transform(img_size)
    val_tf   = get_val_transform(img_size)
    test_tf  = get_test_transform(img_size)

    # ── Datasets ──────────────────────────────────────────────
    train_ds = TN3kDataset(
        image_dir=trainval_img_dir,
        mask_dir=trainval_mask_dir,
        filenames_or_stems=train_stems,
        transform=train_tf,
    )
    val_ds = TN3kDataset(
        image_dir=trainval_img_dir,
        mask_dir=trainval_mask_dir,
        filenames_or_stems=val_stems,
        transform=val_tf,
    )
    test_ds = TN3kDataset(
        image_dir=test_img_dir,
        mask_dir=test_mask_dir,
        filenames_or_stems=None,
        transform=test_tf,
    )

    # ── Loaders ───────────────────────────────────────────────
    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_mem, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_mem, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_mem, drop_last=False,
    )

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────
#  Sanity Check
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.utils import seed_everything, get_device

    seed_everything(42)
    device = get_device()

    print("\n[INFO] Building data loaders ...")
    train_loader, val_loader, test_loader = get_loaders(
        data_root="data/tn3k", batch_size=16, num_workers=2,
    )
    print(f"  Train batches : {len(train_loader)}  ({len(train_loader.dataset)} samples)")
    print(f"  Val   batches : {len(val_loader)}  ({len(val_loader.dataset)} samples)")
    print(f"  Test  batches : {len(test_loader)}  ({len(test_loader.dataset)} samples)")

    # 1. Verify single dataset item return
    train_ds = train_loader.dataset
    single_img, single_mask = train_ds[0]
    print(f"\n[SANITY CHECK] Single sample:")
    print(f"  Single image shape : {single_img.shape}")
    print(f"  Single mask shape  : {single_mask.shape}")
    print(f"  Single mask unique : {single_mask.unique().tolist()}")

    assert single_img.shape == (3, 256, 256), (
        f"Expected single image shape (3, 256, 256), got {single_img.shape}"
    )
    assert single_mask.shape == (1, 256, 256), (
        f"Expected single mask shape (1, 256, 256), got {single_mask.shape}"
    )
    single_mask_vals = set(single_mask.unique().tolist())
    assert single_mask_vals.issubset({0.0, 1.0}), (
        f"Single mask values must be strictly within {{0.0, 1.0}}, got {single_mask_vals}"
    )

    # 2. Verify DataLoader mini-batch
    images, masks = next(iter(train_loader))
    print(f"\n[SANITY CHECK] Mini-batch (B=16):")
    print(f"  Image batch shape  : {images.shape}")
    print(f"  Mask batch shape   : {masks.shape}")
    print(f"  Image dtype        : {images.dtype}")
    print(f"  Mask dtype         : {masks.dtype}")
    print(f"  Image range        : [{images.min():.4f}, {images.max():.4f}]")
    print(f"  Mask unique        : {masks.unique().tolist()}")

    assert images.shape == (16, 3, 256, 256), (
        f"Expected batch image shape (16, 3, 256, 256), got {images.shape}"
    )
    assert masks.shape == (16, 1, 256, 256), (
        f"Expected batch mask shape (16, 1, 256, 256), got {masks.shape}"
    )
    mask_vals = set(masks.unique().tolist())
    assert mask_vals.issubset({0.0, 1.0}), (
        f"Batch mask values must be strictly within {{0.0, 1.0}}, got {mask_vals}"
    )

    print("\n=== ALL ASSERTIONS PASSED ===")

