import sys, os
sys.path.insert(0, os.path.abspath("."))
import time
import torch
from src.dataset import get_loaders
from src.proposed import ResAttentionUNet
from src.metrics import BCEDiceLoss
from src.utils import seed_everything
from torch.optim import AdamW

seed_everything(42)
device = torch.device("cpu")
print("Loading data...")
train_loader, val_loader, _ = get_loaders("data/tn3k", batch_size=16, num_workers=0)

model = ResAttentionUNet(in_channels=3, out_channels=1, attention=True).to(device)
ckpt = torch.load("checkpoints/best_proposed.pth", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
print("Loaded best_proposed at epoch", ckpt["epoch"], "with val_dice:", ckpt["val_dice"])

optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
criterion = BCEDiceLoss()

try:
    print(f"Total train batches: {len(train_loader)}")
    t0 = time.time()
    model.train()
    for i, (imgs, msks) in enumerate(train_loader):
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, msks)
        loss.backward()
        optimizer.step()
        if (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            print(f"Batch [{i+1}/{len(train_loader)}] | Elapsed: {elapsed:.1f}s")
        if i >= 9:
            break
    print("Done in", time.time() - t0, "s")
except Exception as err:
    import traceback
    traceback.print_exc()
