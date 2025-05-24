import torch
from models.deeplob import Deeplob
from pathlib import Path
from data_loader.FI2010 import Dataset_fi2010

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



net = load_deeplob(device="cpu")        # already in eval() mode
print(summary(net, input_size=(1, 1, 20, 40)))  # (B,C,T,D)
w = net.conv1[0].weight.detach().cpu().flatten().numpy()
fig = px.histogram(w, nbins=100, title="Conv-1 weight distribution")
fig.show()
