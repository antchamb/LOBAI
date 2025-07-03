from networkx import set_node_attributes

from data_loader.FI2010 import *
from data_loader.FI2010 import __extract_stock__, __get_raw__, __split_x_y__

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

for i in range(X.shape[0] - 1):
    first_price = (X[i][0][0] + X[i][0][2]) / 2 # mid price


    graphs = [
        snapshot_to_graph(X[i][t], first_price_ref=first_price)
        for t in range(99)
    ]


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

from torch.utils.data import Dataset

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

        return graphs, torch.tensor(label, dtype=torch.long)


# graph_ds = GraphWindowDataset()