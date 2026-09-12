"""
proposed.py — Custom Res-Attention U-Net Architecture
======================================================
Track 2B: Thyroid Nodule Ultrasound Segmentation (TN3k)

Architecture Overview
---------------------
An enhanced U-Net that augments the vanilla baseline with two key additions:

1. **Residual Blocks (`ResBlock`)**
   Replaces standard consecutive convolutions with residual convolutional
   blocks equipped with an identity or 1x1 projection shortcut. This solves the
   vanishing gradient degradation problem across deep feature hierarchies.

2. **Attention Gates (`AttentionGate`)**
   Applies spatial gating before skip concatenation:
   - Uses the coarser decoder feature map as gating signal `g`.
   - Uses encoder skip features `x`.
   - Computes spatial attention coefficients via 1x1 convolutions, additive
     fusion, and sigmoid activation.
   - Suppresses acoustic speckle noise, shadow artifacts, and background
     thyroid parenchyma while accentuating nodule boundaries.

3. **Ablation Study Support**
   The architecture accepts `attention: bool = True`. When set to `False`,
   the network functions as a pure Residual U-Net (Res-UNet), maintaining identical
   channel dimensions and tensor interfaces for controlled ablation benchmarking.

Expected Public API
-------------------
    class AttentionGate(nn.Module):
        def __init__(self, F_g, F_l, F_int=None): ...
        def forward(self, g, x): ...

    class ResBlock(nn.Module):
        def __init__(self, in_ch, out_ch): ...
        def forward(self, x): ...

    class ResAttentionUNet(nn.Module):
        def __init__(self, in_channels=3, out_channels=1, attention=True): ...
        def forward(self, x): ...
"""

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual Convolutional Block.

    Applies two consecutive 3x3 Conv-BatchNorm-ReLU layers with a residual skip
    connection. If in_ch != out_ch, a 1x1 projection convolution with BatchNorm
    is applied to match dimensions.

    Parameters
    ----------
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of output channels.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)

        if in_ch == out_ch:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual addition."""
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual
        return self.relu2(out)


class AttentionGate(nn.Module):
    """Additive Spatial Attention Gate for Medical Ultrasound Segmentation.

    Re-weights the encoder skip connection feature map `x` using the coarse
    gating signal `g` from the deeper decoder stage to suppress speckle noise
    and background artifacts:
        psi = ReLU(W_g(g) + W_x(x))
        alpha = Sigmoid(psi_conv(psi))
        output = x * alpha

    Parameters
    ----------
    F_g : int
        Channel count of the gating signal tensor `g` (from decoder).
    F_l : int
        Channel count of the skip connection tensor `x` (from encoder).
    F_int : int, optional
        Intermediate channel dimension for linear projections.
        Defaults to F_l // 2 (or minimum 1).
    """

    def __init__(self, F_g: int, F_l: int, F_int: Optional[int] = None):
        super().__init__()
        if F_int is None:
            F_int = max(F_l // 2, 1)

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass applying spatial attention coefficients.

        Parameters
        ----------
        g : torch.Tensor
            Gating signal from decoder (B, F_g, H_g, W_g).
        x : torch.Tensor
            Skip feature map from encoder (B, F_l, H_x, W_x).

        Returns
        -------
        torch.Tensor
            Attended skip feature map of shape (B, F_l, H_x, W_x).
        """
        # Ensure spatial dimensions match before projection
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=True)

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi_act = self.relu(g1 + x1)
        alpha = self.psi(psi_act)

        return x * alpha


class DecoderBlock(nn.Module):
    """Decoder stage with Bilinear Upsampling, Attention Gate, and ResBlock.

    Avoids checkerboard deconvolution artifacts by using Bilinear Interpolation
    followed by a 1x1 convolution for channel adjustment before concatenation.

    Parameters
    ----------
    in_channels : int
        Channels in the lower decoder feature map.
    skip_channels : int
        Channels in the corresponding encoder skip connection.
    out_channels : int
        Channels produced after the decoder residual block.
    attention : bool, default True
        Whether to filter skip features through an AttentionGate.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        attention: bool = True,
    ):
        super().__init__()
        self.attention = attention

        # 1x1 Conv + BN + ReLU to adjust upsampled features
        self.up_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        if self.attention:
            self.att_gate = AttentionGate(
                F_g=out_channels,
                F_l=skip_channels,
                F_int=max(skip_channels // 2, 1),
            )
        else:
            self.att_gate = None

        # Post-concat processing through ResBlock
        concat_channels = out_channels + skip_channels
        self.res_block = ResBlock(concat_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Upsample `x`, apply attention on `skip`, concatenate, and process."""
        # Bilinear upsampling to match skip spatial dimensions
        x_up = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        x_up = self.up_conv(x_up)

        if self.attention and self.att_gate is not None:
            skip_filtered = self.att_gate(g=x_up, x=skip)
        else:
            skip_filtered = skip

        out = torch.cat([x_up, skip_filtered], dim=1)
        return self.res_block(out)


class ResAttentionUNet(nn.Module):
    """Custom Residual Attention U-Net for Thyroid Ultrasound Segmentation.

    Constructed layer-by-layer:
    - 4-stage Residual Encoder path with MaxPool downsampling.
    - Central Residual Bottleneck.
    - 4-stage Decoder path with Bilinear Upsampling + 1x1 Convs and Attention Gates.
    - Single 1x1 Conv segmentation head producing raw logits.
    - `attention=True/False` toggle for clean ablation studies.

    Parameters
    ----------
    in_channels : int, default 3
        Number of input channels (e.g., 3 for RGB ultrasound frames).
    out_channels : int, default 1
        Number of segmentation output channels (1 for binary nodule masks).
    features : tuple of int, default (64, 128, 256, 512)
        Channel dimensions for each encoder stage.
    attention : bool, default True
        If True, applies Attention Gates in the decoder skip connections.
        If False, runs as a pure Res-UNet ablation baseline.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: Sequence[int] = (64, 128, 256, 512),
        attention: bool = True,
    ):
        super().__init__()
        self.attention = attention
        self.features = list(features)

        # Encoder Path
        self.enc1 = ResBlock(in_channels, self.features[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = ResBlock(self.features[0], self.features[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = ResBlock(self.features[1], self.features[2])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = ResBlock(self.features[2], self.features[3])
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        bottleneck_channels = self.features[3] * 2
        self.bottleneck = ResBlock(self.features[3], bottleneck_channels)

        # Decoder Path
        self.dec4 = DecoderBlock(
            in_channels=bottleneck_channels,
            skip_channels=self.features[3],
            out_channels=self.features[3],
            attention=attention,
        )
        self.dec3 = DecoderBlock(
            in_channels=self.features[3],
            skip_channels=self.features[2],
            out_channels=self.features[2],
            attention=attention,
        )
        self.dec2 = DecoderBlock(
            in_channels=self.features[2],
            skip_channels=self.features[1],
            out_channels=self.features[1],
            attention=attention,
        )
        self.dec1 = DecoderBlock(
            in_channels=self.features[1],
            skip_channels=self.features[0],
            out_channels=self.features[0],
            attention=attention,
        )

        # Segmentation Head (raw logits)
        self.final_conv = nn.Conv2d(self.features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning unnormalized logits of shape (B, out_channels, H, W)."""
        # Encoder
        s1 = self.enc1(x)
        p1 = self.pool1(s1)

        s2 = self.enc2(p1)
        p2 = self.pool2(s2)

        s3 = self.enc3(p2)
        p3 = self.pool3(s3)

        s4 = self.enc4(p3)
        p4 = self.pool4(s4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder with skip connections
        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        return self.final_conv(d1)


# Alias to support both naming conventions
CustomAttentionUNet = ResAttentionUNet