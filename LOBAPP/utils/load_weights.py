import torch
from models.deeplob import Deeplob
from pathlib import Path
from data_loader.FI2010 import Dataset_fi2010
import random

CKPT = Path(__file__).resolve().parents[1] / "weights" / "deeplob_light.pt"


def load_deeplob(device="cuda" if torch.cuda.is_available() else "cpu"):
    """Load Deeplob-light weights and return an eval-mode model."""
    model = Deeplob(lighten=True).to(device)

    state = torch.load(CKPT, map_location=device)

    # strip "module." if saved with DataParallel
    if next(iter(state)).startswith("module."):
        state = {k.replace("module.", ""): v for k, v in state.items()}

    model.load_state_dict(state, strict=False)
    model.eval()
    return model


import torch, numpy as np, plotly.express as px, plotly.graph_objects as go
from torchinfo import summary                 # pip install torchinfo
from captum.attr import Saliency, IntegratedGradients  # pip install captum
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
import seaborn as sns, matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"
net = load_deeplob(device="cpu")
print(summary(net, input_size=(1, 1, 20, 40)))  # (B,C,T,D)

conv_blocks = [net.conv1, net.conv2, net.conv3, net.inp1, net.inp2, net.inp3]

for i, block in enumerate(conv_blocks, start=1):
    # Check if the first layer in the block has weights
    if hasattr(block[0], "weight"):
        weights = block[0].weight.detach().cpu().flatten().numpy()
        fig = px.histogram(weights, nbins=100, title=f"Conv Block {i} Weight Distribution")
        fig.show()

# Plot weight distributions for LSTM layer
lstm_weights_ih = net.lstm.weight_ih_l0.detach().cpu().flatten().numpy()
fig_ih = px.histogram(lstm_weights_ih, nbins=100, title="LSTM Input-Hidden Weight Distribution")
fig_ih.show()

lstm_weights_hh = net.lstm.weight_hh_l0.detach().cpu().flatten().numpy()
fig_hh = px.histogram(lstm_weights_hh, nbins=100, title="LSTM Hidden-Hidden Weight Distribution")
fig_hh.show()

ds = Dataset_fi2010(
    auction=False,
    normalization="Zscore",
    stock_idx=[0, 1, 2, 3, 4],  # all five stocks
    days=[2],  # day 1 = training set; 2-10 = test sets
    T=20,
    k=0,
    lighten=True,
)
loader = torch.utils.data.DataLoader(ds, batch_size=1024)
all_pred, all_true =  [], []
with torch.no_grad():
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        probs =  net(xb).argmax(1)
        all_pred.append(probs.cpu())
        all_true.append(yb.cpu())

y_pred = torch.cat(all_pred).numpy()
y_true = torch.cat(all_true).numpy()

# Classification report
print(classification_report(y_true, y_pred, target_names=["down", "flat", "up"]))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["down", "flat", "up"],
            yticklabels=["down", "flat", "up"])
plt.title("Confusion Matrix - k = 1")
plt.show()

# Integrated Gradients
ig = IntegratedGradients(net)

# Ensure the input tensor is on the same device as the model
sample_x, _ = ds[random.randint(0, len(ds) - 1)]
sample_x = sample_x.unsqueeze(0).to(device)  # Add batch dimension and move to the correct device

# Compute attributions for the target class (e.g., 'up' = 2)
# Temporarily set the model to training mode
net.train()

# Compute attributions for the target class (e.g., 'up' = 2)
attr, _ = ig.attribute(sample_x, target=2, return_convergence_delta=True)

# Switch the model back to evaluation mode
net.eval()

# Process and visualize attributions
attr = attr.squeeze(0).squeeze(0).cpu().numpy()  # Move to CPU for visualization
plt.figure(figsize=(5, 4))
sns.heatmap(attr, cmap="coolwarm", center=0)
plt.title("Integrated Gradients – label ‘up’")
plt.xlabel("Feature index (40)")
plt.ylabel("Time steps (20)")
plt.show()

def occlude_feature(x, feat_idx):
    x2 = x.clone()
    x2[:,:,feat_idx] = 0
    return x2

batch_x, _ = next(iter(loader))[:128]   # small batch
base_prob = net(batch_x).softmax(1)[:,2]  # probability of 'up'

impact = []
for f in range(batch_x.shape[-1]):      # 40 features
    prob = net(occlude_feature(batch_x.clone(), f)).softmax(1)[:,2]
    impact.append((base_prob - prob).abs().mean().item())

fig = go.Figure(go.Bar(x=list(range(40)), y=impact))
fig.update_layout(title="Average drop in P(up) when occluding each feature",
                  xaxis_title="Feature idx", yaxis_title="Δ probability")
fig.show()
