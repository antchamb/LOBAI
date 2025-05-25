from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from models import Deeplob
from data_loader.FI2010 import *   # <- rename if needed
import multiprocessing as mp
from tqdm import tqdm




print("CUDA visible :", torch.cuda.is_available())          # → True
print("Selected GPU :", torch.cuda.get_device_name(0))       # → NVIDIA RTX A2000 12GB

# ------------ CONFIG ------------
LIGHTEN      = True          # False for 40-feature version
T_WINDOW     = 100           # 100 is standard for FI-2010
K_HORIZON    = 2             # 0,1,2,3,4 => 10/20/30/40/50-tick horizons
BATCH_SIZE   = 64
EPOCHS       = 10
LR           = 1e-4
DATA_ROOT    = r"D:\Bureau\FI-2010"     # your absolute path

CKPT_DIR     = Path(__file__).parent.parent / "weights"
CKPT_DIR.mkdir(exist_ok=True)
CKPT_FILE    = CKPT_DIR / f"deeplob_{'light' if LIGHTEN else 'full'}.pt"

print("a")
# ------------ DATASET ------------
train_set = Dataset_fi2010(
    auction=False,
    normalization="Zscore",
    stock_idx=[0, 1, 2, 3, 4],   # all five stocks
    days=[1,2,3,4,5,6,7,8,9,10], # day 1 = training set; 2-10 = test sets
    T=T_WINDOW,
    k=K_HORIZON,
    lighten=LIGHTEN,
)
print("b")
# ------------ DATALOADER (Windows notes) ------------
NUM_WORKERS = 0  # set to 4 on Linux/macOS; keep 0 on Windows unless you add the __main__ guard
train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)
print("c")

# sanity-check one batch
xb, yb = next(iter(train_loader))
assert xb.shape[1:] == (1, T_WINDOW, 20 if LIGHTEN else 40), \
       f"shape mismatch — got {xb.shape}"

print("d")

# ------------ MODEL ------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model  = Deeplob(lighten=LIGHTEN).to(device)
opt    = torch.optim.Adam(model.parameters(), lr=LR)
crit   = nn.CrossEntropyLoss()
print("e")

# ------------ TRAIN ------------
for epoch in tqdm(range(1, EPOCHS + 1)):
    model.train()
    running, total = 0.0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward()
        opt.step()
        running += loss.item() * yb.size(0)
        total   += yb.size(0)
    print(f"epoch {epoch:02d} | loss {running/total:.4f}")

CKPT_FILE    = CKPT_DIR / f"test.pt"
# ------------ SAVE ------------
torch.save(model.state_dict(), CKPT_FILE)
print("✔ checkpoint saved →", CKPT_FILE)

# ------------ Windows entry-point guard ------------
if __name__ == "__main__":
    mp.freeze_support()   # no-op on Linux; required for pyinstaller on Windows
