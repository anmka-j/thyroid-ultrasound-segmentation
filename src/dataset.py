"""
dataset.py — TN3k Dataset Loader
=================================
Track 2B: Thyroid Nodule Ultrasound Segmentation (TN3k)

Responsibilities
----------------
- Load thyroid-nodule ultrasound images and their corresponding binary masks
  from the TN3k dataset directory structure.
- Resize all images and masks to 256×256 pixels.
- Apply channel-wise normalization (ImageNet statistics or dataset-specific).
- Provide an official 90 / 10 train / validation split with reproducible
  shuffling (seeded via `utils.seed_everything`).
- Support optional Albumentations-based augmentation pipelines passed at
  construction time.

Expected public API
-------------------
    class TN3kDataset(torch.utils.data.Dataset):
        def __init__(self, image_dir, mask_dir, transform=None): ...
        def __len__(self): ...
        def __getitem__(self, idx): ...

    def get_loaders(data_root, batch_size, num_workers, transform_train, transform_val): ...
"""
