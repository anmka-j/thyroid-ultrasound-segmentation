"""
proposed.py — Custom Res-Attention U-Net Architecture
======================================================
Track 2B: Thyroid Nodule Ultrasound Segmentation (TN3k)

Architecture Overview
---------------------
An enhanced U-Net that augments the vanilla baseline with two key additions:

1. **Residual Encoder Blocks**
   Each encoder stage replaces plain Conv-BN-ReLU pairs with residual blocks
   that include a learnable shortcut (1×1 Conv when channel dims change),
   enabling deeper gradient flow and faster convergence.

2. **Attention Gates in the Decoder**
   Before each skip-connection concatenation, an Attention Gate re-weights
   the encoder feature map using the gating signal from the decoder,
   suppressing irrelevant background regions and focusing on the nodule.

Encoder backbone may optionally leverage a pre-trained `timm` model
(e.g., ResNet-34) for transfer learning.

Expected public API
-------------------
    class AttentionGate(nn.Module):
        def __init__(self, F_g, F_l, F_int): ...
        def forward(self, g, x): ...

    class ResBlock(nn.Module):
        def __init__(self, in_ch, out_ch): ...
        def forward(self, x): ...

    class ResAttentionUNet(nn.Module):
        def __init__(self, in_channels=3, out_channels=1): ...
        def forward(self, x): ...
"""
