from pathlib import Path

import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader.FI2010 import Dataset_fi2010
from models.translob import TransLOB


print("CUDA visible :", torch.cuda.is_available())  # → True
print("Selected GPU :", torch.cuda.get_device_name(0))       # → NVIDIA RTX A2000 12GB

# ------------ CONFIG ------------
LIGHTEN    = False          # False for 40-feature version
T_WINDOW   = 100            # 100 is standard for FI-2010
K_HORIZON  = 4              # 0,1,2,3,4 => 10/20/30/40/50-tick horizons
BATCH_SIZE = 32
EPOCHS     = 150
LR         = 1e-4
DATA_ROOT  = r"D:\Bureau\FI-2010"

CKPT_DIR     = Path(__file__).parent.parent / "weights"
CKPT_DIR.mkdir(exist_ok=True)
CKPT_FILE    = CKPT_DIR / f"translob_test_10.pt"
print(CKPT_FILE)
train_set = Dataset_fi2010(
    auction=False,
    normalization="Zscore",
    stock_idx=[0, 1, 2, 3, 4],   # all five stocks
    days=[1,2,3,4,5,6,7],
    T=T_WINDOW,
    k=K_HORIZON,
    lighten=LIGHTEN,
)


NUM_WORKERS = 0
train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

class_counts = np.zeros(3, dtype=np.int64)
for _, yb in train_loader:
    for c in range(3):
        class_counts[c] += (yb == c).sum().item()

class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
class_weights /= class_weights.sum() * 3


# verify shape (B, 1, T, 40)
xb0, yb0 = next(iter(train_loader))
assert xb0.shape[1:] == (1, T_WINDOW, 40), f"got {xb0.shape}"

# Model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_weights = class_weights.to(device)
model  = TransLOB().to(device)
opt    = torch.optim.Adam(model.parameters(), lr=LR)
crit   = nn.CrossEntropyLoss(weight = class_weights)

# Training loop
for epoch in tqdm(range(1, EPOCHS + 1), desc="Training Epochs"):
    model.train()
    running, total = 0.0, 0
    for xb, yb in train_loader:
        xb, yb = xb.squeeze(1).to(device), yb.to(device)
        opt.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward()
        opt.step()
        running += loss.item() * yb.size(0)
        total   += yb.size(0)
    print(f"epoch {epoch:03d} | loss {running/total: .4f}")

# Save the model
torch.save(model.state_dict(), CKPT_FILE)
