from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader.FI2010 import Dataset_fi2010
from models.translob import TransLOB
from training.train_deeplob import DATA_ROOT, BATCH_SIZE

print("CUDA visible :", torch.cuda.is_available())  # → True
print("Selected GPU :", torch.cuda.get_device_name(0))       # → NVIDIA RTX A2000 12GB

# ------------ CONFIG ------------
LIGHTEN    = False          # False for 40-feature version
T_WINDOW   = 100            # 100 is standard for FI-2010
K_HORIZON  = 2              # 0,1,2,3,4 => 10/20/30/40/50-tick horizons
BATCH_SIZE = 32
EPOCHS     = 150
LR         = 1e-4
DATA_ROOT  = r"D:\Bureau\FI-2010"

CKPT_DIR     = Path(__file__).parent.parent / "weights"
CKPT_DIR.mkdir(exist_ok=True)
CKPT_FILE    = CKPT_DIR / f"trans_lob_test.pt"

train_set = Dataset_fi2010(
    auction=False,
    normalization="Zscore",
    stock_idx=[0, 1, 2, 3, 4],   # all five stocks
    days=[1,2,3,4,5,6,7], # day 1 = training set; 2-10 = test sets
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

# verify shape (B, 1, T, 40)
xb0, yb0 = next(iter(train_loader))
assert xb0.shape[1:] == (1, T_WINDOW, 40), f"got {xb0.shape}"

# Model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model  = TransLOB().to(device)
opt    = torch.optim.Adam(model.parameters(), lr=LR)
crit   = nn.CrossEntropyLoss()

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
