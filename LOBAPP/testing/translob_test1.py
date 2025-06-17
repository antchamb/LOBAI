from pathlib import Path
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from data_loader.FI2010 import Dataset_fi2010
from models.translob import TransLOB


CKPT =  Path(__file__).resolve().parents[1] / "weights" / "trans_lob_test.pt"

# load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = TransLOB().to(device)

model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()

test_set = Dataset_fi2010(
    auction=False,
    normalization="Zscore",
    stock_idx=[0, 1, 2, 3, 4],   # all five stocks
    days=[8,9,10], # last 3 days
    T=100,
    k=2,
    lighten=False,
)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

# Evaluate the model
all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb).argmax(dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(yb.cpu())

# compute accuracy
all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)
accuracy = accuracy_score(all_labels, all_preds)
print(f"Accuracy on the last 3 days: {accuracy:.4f}")
