

from data_loader.FI2010 import *



X, y = Dataset_fi2010(
    auction       = False,
    normalization = 'Zscore',
    stock_idx     = [0],
    days          = [1],
    T             = 100,
    k             = [3],
    lighten       = False
).__init_dataset__()



import torch
from torch_geometric.data import Data


def build_edge_index(num_levels: int = 10, cross_edges: bool = True) -> torch.Tensor:
    """
    Génère la liste des arêtes (i,j) et la retourne au format (2, E).
    - intra-bid  : i ↔ i+1   pour i = 0..8
    - intra-ask  : (i+10) ↔ (i+11)
    - cross-side : i ↔ i+10  (si cross_edges=True)
    """
    edges = []

    # 1) voisinage séquentiel bid et ask
    for i in range(num_levels - 1):
        # bid
        edges.append((i, i + 1))
        edges.append((i + 1, i))
        # ask (offset = +num_levels)
        a = i + num_levels
        edges.append((a, a + 1))
        edges.append((a + 1, a))

    # 2) arêtes bid ↔ ask niveau par niveau
    if cross_edges:
        for i in range(num_levels):
            edges.append((i, i + num_levels))
            edges.append((i + num_levels, i))

    # 3) tensor final (2, E)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


def snapshot_to_graph(snapshot: np.ndarray,
                      *, first_price_ref=None,
                      dtype=torch.float32,
                      cross_edges=True) -> Data:
    # --- 1. reshape sans copie -----------------------------
    snap = snapshot.reshape(10, 4)          # (levels, [pA vA pB vB])

    # --- 2. split colonnes ---------------------------------
    ask_p, ask_v, bid_p, bid_v = snap.T     # chacun (10,)

    # --- 3. normalisation ---------------------------------
    if first_price_ref is None:
        first_price_ref = bid_p[0]          # prix bid_1
    bid_p = bid_p / first_price_ref
    ask_p = ask_p / first_price_ref

    # --- 4. construire nœuds -------------------------------
    bid_nodes = np.stack([bid_p, bid_v, np.zeros(10)], axis=1)  # side=0
    ask_nodes = np.stack([ask_p, ask_v, np.ones(10)],  axis=1)  # side=1
    x = np.concatenate([bid_nodes, ask_nodes], axis=0)          # (20,3)
    x = torch.tensor(x, dtype=dtype)

    # --- 5. arêtes (identiques à avant) --------------------
    edge_index = build_edge_index(cross_edges=cross_edges)

    return Data(x=x, edge_index=edge_index)

# for i in range(X.shape[0] - 1):
#     first_price = (X[i][0][0] + X[i][0][2]) / 2 # mid price
#
#
#     graphs = [
#         snapshot_to_graph(X[i][t], first_price_ref=first_price)
#         for t in range(99)
#     ]


# data = snapshot_to_graph(snap)

import networkx as nx
import matplotlib.pyplot as plt

# Convert edge_index to a NetworkX graph
def visualize_graph(data):
    edge_index = data.edge_index.numpy()
    G = nx.Graph()
    G.add_edges_from(edge_index.T)  # Add edges from edge_index

    # Plot the graph
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)  # Layout for better visualization
    nx.draw(
        G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10
    )
    plt.title("Graph Visualization")
    plt.show()

# Visualize the graph
# visualize_graph(data)

from torch.utils.data import Dataset, DataLoader

from torch.utils.data import Dataset
import numpy as np, torch

class GraphWindowDataset(Dataset):
    """
    Transforme les fenêtres FI-2010 *déjà préparées* en
    (liste de graphes, label).
    """
    def __init__(self, fi_ds, *, local_norm=True):
        self.fi_ds = fi_ds
        self.local_norm = local_norm

    def __len__(self):
        return len(self.fi_ds)

    def __getitem__(self, idx):
        window, label = self.fi_ds[idx]      # window:(1,100,40)
        window = window.squeeze(0).numpy()   # (100,40)

        ref = window[0, 2] if self.local_norm else None
        graphs = [snapshot_to_graph(snap, first_price_ref=ref)
                  for snap in window]

        return graphs, label.clone().detach().to(torch.long)


# fi_ds = Dataset_fi2010(
#     auction       = False,
#     normalization = 'Zscore',
#     stock_idx     = [0],
#     days          = [1],
#     T             = 100,
#     k             = 3,
#     lighten       = False
# )
# graph_ds = GraphWindowDataset(fi_ds)
# graph_ld = DataLoader(
#     graph_ds,
#     batch_size=32,
#     shuffle=True,
#     collate_fn=lambda b: (list(zip(*b))[0], torch.stack(list(zip(*b))[1])),
#     num_workers=4,
# )


# Model creation:
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_add_pool

class GraphEncoder(nn.Module):
    def __init__(
            self,
            in_channels = 3,  # price, size, side
            hidden_dim  = 64, # taille embedding noeud
            num_layers  = 2,
            heads       = 4,
            dropout     = 0.1):
        super().__init__()
        self.convs = nn.ModuleList([
            GATConv(
                in_channels if i == 0 else hidden_dim, # input
                hidden_dim, # output
                heads=heads,  # nombre de têtes d'attention
                concat=False,
                dropout=dropout
            )
            for i in range(num_layers)
        ])
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, batch: Batch):
        x, edge_index = batch.x, batch.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.act(x)
            x = self.drop(x)
        g_vec = global_add_pool(x, batch.batch)
        return g_vec

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)                # (L,64)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                        * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)   #  **← produit, pas division**
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)       # (L,64)

    def forward(self, x):
        # x : (seq_len, batch, d_model)
        pe = self.pe[:x.size(0)].unsqueeze(1)   # (seq_len, 1, 64)
        return x + pe                          # broadcast sur batch


def _make_transformer(d_model=64, nhead=4, ffn=256, dropout=0.1, layers=2):
    enc_layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=nhead,
        dim_feedforward=ffn, dropout=dropout,
        activation='gelu', batch_first=False, norm_first=True)
    return nn.TransformerEncoder(enc_layer, num_layers=layers)

class GraphTemporalTransformer(nn.Module):
    def __init__(self,
                 gnn_hidden=64,
                 gnn_layers=2,
                 gnn_heads=4,
                 transformer_layers=2,
                 nhead=4,
                 dim_feedforward=256,
                 dropout=0.1,
                 num_classes=3):
        super().__init__()

        # 4-a  encodeur spatial
        self.graph_enc = GraphEncoder(
            in_channels=3,
            hidden_dim=gnn_hidden,
            num_layers=gnn_layers,
            heads=gnn_heads,
            dropout=dropout)

        # 4-b  encodeur temporel
        self.pos_enc   = PositionalEncoding(gnn_hidden)
        self.transform = _make_transformer(
            d_model=gnn_hidden,
            nhead=nhead,
            ffn=dim_feedforward,
            dropout=dropout,
            layers=transformer_layers)

        # 4-c  tête de classification
        self.cls_head = nn.Sequential(
            nn.LayerNorm(gnn_hidden),
            nn.Linear(gnn_hidden, num_classes)
        )

    # helper : encode une séquence (List[Data]) → (seq_len, 1, H)
    def _encode_seq(self, seq):
        batch = Batch.from_data_list(seq)          # merge graphs
        g_vec = self.graph_enc(batch)              # (T, H)
        return g_vec.unsqueeze(1)                  # + dim batch

    def forward(self, sequences):
        # sequences : List[ List[Data] ]  (batch d’items)
        seq_embeds = [self._encode_seq(seq) for seq in sequences]
        max_len = max(e.size(0) for e in seq_embeds)

        # padding 0 pour aligner
        padded = []
        for e in seq_embeds:
            pad_len = max_len - e.size(0)
            if pad_len:
                e = torch.cat([e, torch.zeros(pad_len, 1, e.size(2),
                                               device=e.device)], 0)
            padded.append(e)
        x = torch.cat(padded, 1)        # (seq_len, batch, H)

        x = self.pos_enc(x)
        h = self.transform(x)           # même shape
        h_last = h[-1]                  # (batch, H)

        return {"logits": self.cls_head(h_last)}

    # perte composite (ici juste CE)
    def compute_loss(self, out, y, lambda_c=1.0):
        return F.cross_entropy(out["logits"], y) * lambda_c


# ENTRAINEMENT

from pathlib import Path

DATA_ROOT = r"D:\Bureau\FI-2010"
CKPT_DIR     = Path(__file__).parent.parent / "weights"
CKPT_DIR.mkdir(exist_ok=True)
CKPT_FILE    = CKPT_DIR / f"gttlob_k10_7days.pt"
print(CKPT_FILE)


train_set = Dataset_fi2010(
    auction       = False,
    normalization = 'Zscore',
    stock_idx     = [0, 1, 2, 3, 4],
    days          = [1, 2, 3, 4, 5, 6, 7],
    T             = 100,
    k             = 4,
    lighten       = False
)

graph_ds = GraphWindowDataset(train_set, local_norm=True)
train_loader = DataLoader(
    graph_ds,
    batch_size=32,
    shuffle=True,
    collate_fn=lambda b: (list(zip(*b))[0], torch.stack(list(zip(*b))[1])),
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

seqs, y = next(iter(train_loader))
print(len(seqs))        # 32
print(len(seqs[0]))     # 100
print(seqs[0][0].x[:2]) # ≠ seqs[0][1].x[:2]
print(y[:5])            # 0/1/2

device = "cuda" if torch.cuda.is_available() else "cpu"
model  = GraphTemporalTransformer().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=3e-4,
                steps_per_epoch=len(train_loader), epochs=100)  # test sur 2 époques

def train_one_epoch(model, loader):
    model.train()
    total_loss = 0
    for seqs, y in loader:
        seqs = [[g.to(device) for g in s] for s in seqs]
        y    = y.to(device)

        out  = model(seqs)
        loss = model.compute_loss(out, y)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)

from tqdm import tqdm

for epoch in tqdm(range(100)):
    loss = train_one_epoch(model, train_loader)
    print(f"Epoch {epoch}  loss={loss:.4f}")


def collate_graph_sequences(batch):
    seqs, labels = zip(*batch)               # tuples
    return list(seqs), torch.stack(labels)

# evaluation
eval_loader = DataLoader(
    graph_ds,
    batch_size=64,                   # plus gros batch possible
    shuffle=False,
    collate_fn=collate_graph_sequences,
    num_workers=0
)
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score
import numpy as np, torch

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_pred, all_true = [], []
    for seqs, y in loader:
        seqs = [[g.to(device) for g in s] for s in seqs]
        out  = model(seqs)
        pred = out["logits"].argmax(1).cpu().numpy()
        all_pred.extend(pred)
        all_true.extend(y.numpy())
    all_pred, all_true = np.array(all_pred), np.array(all_true)
    acc = accuracy_score(all_true, all_pred)
    f1  = f1_score(all_true, all_pred, average="macro")
    mcc = matthews_corrcoef(all_true, all_pred)
    return acc, f1, mcc
acc, f1, mcc = evaluate(model, eval_loader)
print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, MCC: {mcc:.4f}")

torch.save(model.state_dict(), CKPT_FILE)

