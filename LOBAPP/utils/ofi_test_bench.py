import torch
from torch.utils.data import DataLoader
from models.deeplob import Deeplob
from data_loader.FI2010 import *
from sklearn.metrics import accuracy_score
from pathlib import Path


CKPT = Path(__file__).resolve().parents[1] / "weights" / "ofi_epoch_150.pt"
# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Deeplob(lighten=False).to(device)
weights_path = Path(r"D:\Bureau\LOBAI\LOBAPP\weights\ofi_epoch_150.pt")
if not weights_path.exists():
    raise FileNotFoundError(f"Weight file not found: {weights_path}")

model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()

# Prepare the dataset for the last 3 days
test_set = OFIDataset(
    auction=False,
    normalization="Zscore",
    stock_idx=[0, 1, 2, 3, 4],  # all five stocks
    days=[8, 9, 10],  # last 3 days
    T=100,
    k=4,
    lighten=False,
)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# Evaluate the model
all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb).argmax(dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(yb.cpu())

# Calculate accuracy
all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)
accuracy = accuracy_score(all_labels, all_preds)
print(f"Accuracy on the last 3 days: {accuracy:.4f}")