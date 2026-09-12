"""
eda.py — Exploratory Data Analysis
====================================
Track 2B: Thyroid Nodule Ultrasound Segmentation (TN3k)

Responsibilities
----------------
- Scan the TN3k dataset directory and report total image/mask counts.
- Compute and visualise image resolution distribution (width × height).
- Analyse **class imbalance**: for each mask, compute the ratio of foreground
  (nodule) pixels to background pixels and plot the distribution.
- Display a grid of sample image–mask overlay pairs for visual sanity checks.
- Report per-channel mean and standard deviation across the training split
  (useful for custom normalisation).

Expected public API
-------------------
    def dataset_summary(data_root): ...
    def plot_class_distribution(mask_dir): ...
    def show_samples(image_dir, mask_dir, n=8): ...
    def compute_channel_stats(image_dir): ...
"""
