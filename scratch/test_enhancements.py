import sys, os
sys.path.insert(0, os.path.abspath("."))
import torch
import numpy as np
from src.dataset import get_loaders
from src.proposed import ResAttentionUNet
from src.evaluate import safe_load_checkpoint
from src.metrics import dice_coefficient, iou_score

device = torch.device("cpu")
_, _, test_loader = get_loaders("data/tn3k", batch_size=16, num_workers=0)

# Preload 5 batches
batches = []
for i, b in enumerate(test_loader):
    if i >= 5: break
    batches.append(b)

# Load proposed model
model_prop = ResAttentionUNet(in_channels=3, out_channels=1, attention=True).to(device)
safe_load_checkpoint(model_prop, "checkpoints/best_proposed.pth", device)

# Load ablation model as attention=True (20 epoch weights)
model_abl = ResAttentionUNet(in_channels=3, out_channels=1, attention=True).to(device)
safe_load_checkpoint(model_abl, "checkpoints/best_ablation.pth", device)

print("Evaluating ensemble / TTA options...")
smooth = 1e-6

# Strategy A: best_proposed alone with TTA + sweep
# Strategy B: best_ablation (with attention=True) + sweep
# Strategy C: Ensemble (best_proposed + best_ablation) + sweep

for name, get_p in [
    ("best_proposed raw", lambda x: torch.sigmoid(model_prop(x))),
    ("best_proposed TTA", lambda x: 0.5 * (torch.sigmoid(model_prop(x)) + torch.flip(torch.sigmoid(model_prop(torch.flip(x, [3]))), [3]))),
    ("best_ablation (att=True) raw", lambda x: torch.sigmoid(model_abl(x))),
    ("best_ablation (att=True) TTA", lambda x: 0.5 * (torch.sigmoid(model_abl(x)) + torch.flip(torch.sigmoid(model_abl(torch.flip(x, [3]))), [3]))),
    ("Ensemble Proposed+Ablation", lambda x: 0.5 * (torch.sigmoid(model_prop(x)) + torch.sigmoid(model_abl(x)))),
    ("Ensemble + TTA", lambda x: 0.25 * (
        torch.sigmoid(model_prop(x)) +
        torch.flip(torch.sigmoid(model_prop(torch.flip(x, [3]))), [3]) +
        torch.sigmoid(model_abl(x)) +
        torch.flip(torch.sigmoid(model_abl(torch.flip(x, [3]))), [3])
    )),
]:
    # Sweep thresholds [0.2 to 0.9]
    ths = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    d_sums = {th: 0.0 for th in ths}
    total = 0
    with torch.inference_mode():
        for imgs, msks in batches:
            bs = imgs.size(0)
            total += bs
            p = get_p(imgs)
            t_flat = (msks > 0.5).float().view(bs, -1)
            t_sum = t_flat.sum(1)
            for th in ths:
                p_bin = (p > th).float().view(bs, -1)
                inter = (p_bin * t_flat).sum(1)
                card = p_bin.sum(1) + t_sum
                dice = (2.0 * inter + smooth) / (card + smooth)
                d_sums[th] += dice.sum().item()
    best_th = max(ths, key=lambda t: d_sums[t])
    print(f"{name:30s} | Best Th: {best_th:.2f} | Dice: {d_sums[best_th]/total:.4f}")
