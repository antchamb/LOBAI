from pathlib import Path
import torch, torch.nn as nn
from models import Deeplob

lighten = True
out_dir = Path(__file__).parent.parent / "weights"
out_dir.mkdir(exist_ok=True)
ckpt_file = out_dir / f"deeplob_{'light' if lighten else 'full'}.pt"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Deeplob(lighten=lighten).to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

train_lodaer
epochs = 5

for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optim.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optim.step()

torch.save(model.stat_dict(), ckpt_file)
print(f"✓ Saved weights → {ckpt_file}")

