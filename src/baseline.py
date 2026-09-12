"""
baseline.py — Vanilla U-Net Baseline Architecture
===================================================
Track 2B: Thyroid Nodule Ultrasound Segmentation (TN3k)

Architecture Overview
---------------------
A standard 4-stage encoder–decoder U-Net:

    Encoder (contracting path)
    ──────────────────────────
    Stage 1:  3  → 64  channels  │  two 3×3 Conv-BN-ReLU blocks + 2×2 MaxPool
    Stage 2: 64  → 128 channels  │  two 3×3 Conv-BN-ReLU blocks + 2×2 MaxPool
    Stage 3: 128 → 256 channels  │  two 3×3 Conv-BN-ReLU blocks + 2×2 MaxPool
    Stage 4: 256 → 512 channels  │  two 3×3 Conv-BN-ReLU blocks + 2×2 MaxPool

    Bottleneck
    ──────────
    512 → 1024 channels          │  two 3×3 Conv-BN-ReLU blocks

    Decoder (expansive path)
    ────────────────────────
    Stage 4: 1024 → 512 channels │  2×2 TransposeConv ↑ + skip-concat + 2× Conv
    Stage 3:  512 → 256 channels │  2×2 TransposeConv ↑ + skip-concat + 2× Conv
    Stage 2:  256 → 128 channels │  2×2 TransposeConv ↑ + skip-concat + 2× Conv
    Stage 1:  128 →  64 channels │  2×2 TransposeConv ↑ + skip-concat + 2× Conv

    Head:  64 → 1 channel        │  1×1 Conv (no activation — raw logits)

Input : (B, 3, 256, 256)
Output: (B, 1, 256, 256)   — raw logits; apply sigmoid for probability map.

Expected public API
-------------------
    class UNet(nn.Module):
        def __init__(self, in_channels=3, out_channels=1): ...
        def forward(self, x): ...
"""
