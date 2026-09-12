"""
baseline.py — Vanilla U-Net Baseline Architecture
===================================================
Track 2B: Thyroid Nodule Ultrasound Segmentation (TN3k)

Architecture Overview
---------------------
A standard 4-stage encoder–decoder convolutional U-Net built from scratch in PyTorch.
Serves as Model 1: The Mandatory Reference Benchmark Floor.

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

    Head
    ────
    64 → 1 channel               │  1×1 Conv (raw unactivated logits)

No pretrained weights, attention modules, or residual shortcuts are used.
"""

import os
import sys
from typing import Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
#  Building Blocks
# ─────────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    """Standard DoubleConv block: [Conv2d(3x3, padding=1) -> BatchNorm2d -> ReLU] x 2."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ─────────────────────────────────────────────────────────────
#  Vanilla U-Net Model
# ─────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """Standard 4-stage Vanilla U-Net for binary thyroid nodule segmentation.

    Parameters
    ----------
    in_channels : int, default=3
        Number of input channels (RGB replicated ultrasound scans).
    out_channels : int, default=1
        Number of output channels (1 for binary segmentation logits).
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # ── Encoder (Contracting Path) ─────────────────────────
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ── Bottleneck ─────────────────────────────────────────
        self.bottleneck = DoubleConv(512, 1024)

        # ── Decoder (Expansive Path) ───────────────────────────
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # ── Segmentation Head ──────────────────────────────────
        self.head = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)               # (B, 64, H, W)
        p1 = self.pool1(e1)             # (B, 64, H/2, W/2)

        e2 = self.enc2(p1)              # (B, 128, H/2, W/2)
        p2 = self.pool2(e2)             # (B, 128, H/4, W/4)

        e3 = self.enc3(p2)              # (B, 256, H/4, W/4)
        p3 = self.pool3(e3)             # (B, 256, H/8, W/8)

        e4 = self.enc4(p3)              # (B, 512, H/8, W/8)
        p4 = self.pool4(e4)             # (B, 512, H/16, W/16)

        # Bottleneck
        b = self.bottleneck(p4)         # (B, 1024, H/16, W/16)

        # Decoder with skip-connections
        d4 = self.up4(b)                # (B, 512, H/8, W/8)
        d4 = self._cat(d4, e4)          # (B, 1024, H/8, W/8)
        d4 = self.dec4(d4)              # (B, 512, H/8, W/8)

        d3 = self.up3(d4)               # (B, 256, H/4, W/4)
        d3 = self._cat(d3, e3)          # (B, 512, H/4, W/4)
        d3 = self.dec3(d3)              # (B, 256, H/4, W/4)

        d2 = self.up2(d3)               # (B, 128, H/2, W/2)
        d2 = self._cat(d2, e2)          # (B, 256, H/2, W/2)
        d2 = self.dec2(d2)              # (B, 128, H/2, W/2)

        d1 = self.up1(d2)               # (B, 64, H, W)
        d1 = self._cat(d1, e1)          # (B, 128, H, W)
        d1 = self.dec1(d1)              # (B, 64, H, W)

        # Output raw unactivated logits
        logits = self.head(d1)          # (B, out_channels, H, W)
        return logits

    @staticmethod
    def _cat(upsampled: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Concatenate upsampled feature map with encoder skip connection along channels."""
        # Handle potential padding discrepancies if spatial dimensions differ
        diff_y = skip.size()[2] - upsampled.size()[2]
        diff_x = skip.size()[3] - upsampled.size()[3]
        if diff_y > 0 or diff_x > 0:
            upsampled = F.pad(
                upsampled,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )
        return torch.cat([skip, upsampled], dim=1)

    @property
    def loss_fn(self) -> nn.BCEWithLogitsLoss:
        """Baseline training criterion: BCEWithLogitsLoss."""
        return nn.BCEWithLogitsLoss()

    def get_criterion(self) -> nn.BCEWithLogitsLoss:
        """Return the baseline training loss criterion."""
        return nn.BCEWithLogitsLoss()


# ─────────────────────────────────────────────────────────────
#  Loss Function Utility
# ─────────────────────────────────────────────────────────────

def get_loss_fn() -> nn.BCEWithLogitsLoss:
    """Return the baseline training criterion for binary segmentation."""
    return nn.BCEWithLogitsLoss()


def count_parameters(model: nn.Module) -> int:
    """Calculate the total number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────
#  Smoke Test Verification
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Vanilla U-Net Baseline (Model 1) -- Architecture Verification")
    print("=" * 60)

    # 1. Instantiate model
    model = UNet(in_channels=3, out_channels=1)
    model.eval()

    # 2. Count parameters
    total_params = count_parameters(model)
    print(f"  Model Name            : Vanilla UNet")
    print(f"  Total Trainable Params: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"  Loss Criterion        : {model.loss_fn.__class__.__name__}")

    # 3. Pass dummy batch tensor (2, 3, 256, 256)
    dummy_input = torch.randn(2, 3, 256, 256)
    print(f"\n  [Input Tensor]        : {dummy_input.shape}")

    with torch.no_grad():
        output_logits = model(dummy_input)

    print(f"  [Output Logits]       : {output_logits.shape}")
    print(f"  [Output Dtype]        : {output_logits.dtype}")
    print(f"  [Output Range]        : [{output_logits.min().item():.4f}, {output_logits.max().item():.4f}]")

    # 4. Assertions
    expected_shape = (2, 1, 256, 256)
    assert output_logits.shape == expected_shape, (
        f"Assertion Failed! Expected output shape {expected_shape}, but got {output_logits.shape}"
    )

    # 5. Loss calculation sanity check
    dummy_targets = torch.randint(0, 2, (2, 1, 256, 256)).float()
    criterion = model.get_criterion()
    loss = criterion(output_logits, dummy_targets)
    print(f"  [Dummy Batch Loss]    : {loss.item():.4f}")

    assert not torch.isnan(loss), "Loss computation produced NaN!"
    assert not torch.isinf(loss), "Loss computation produced Inf!"

    print("\n=== ALL ASSERTIONS & SMOKE TESTS PASSED ===")
    print("=" * 60)
