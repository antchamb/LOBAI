#
# import dash
# import dash_bootstrap_components as dbc
# from dash import html, dcc
#
#
#
#
# app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
#
#
# header = html.Div([
#     html.Div([
#         html.Div([
#             dcc.Link(
#                 page['name'] + ' | ',
#                 href=page['path'],
#                 style={
#                     'color': '#fff',
#                     'fontSize': '18px',
#                     'padding': '0 10px',
#                     'textDecoration': 'none',
#                 }
#             )
#             for page in dash.page_registry.values()
#         ], style={
#             'display': 'flex',
#             'justifyContent': 'center',
#             'alignItems': 'center',
#             'height': '100%',
#         }),
#     ], style={
#         'display': 'flex',
#         'flexDirection': 'column',
#         'justifyContent': 'center',
#         'alignItems': 'center',
#         'padding': '0',
#         'height': '100%',
#     })
# ], style={
#     'backgroundColor': '#1a1a1a',
#     'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
#     'width': '100%',
#     'position': 'fixed',
#     'top': '0',
#     'left': '0',
#     'height': '7vh',
#     'zIndex': '999',
#     'margin': '0',
#     'padding': '0'
# })
#
# # force html structure
# app.index_string = '''
# <!DOCTYPE html>
#
# <html>
#     <head>
#         {%metas%}
#         <title>{%title%}</title>
#         {%favicon%}
#         {%css%}
#         <style>
#             * {
#                 margin: 0;
#                 padding: 0;
#                 box-sizing: border-box;
#             }
#             html, body {
#                 width: 100%;
#                 height: 100%;
#                 margin: 0;
#                 padding: 0;
#             }
#         </style>
#     </head>
#     <body>
#         {%app_entry%}
#         <footer>
#             {%config%}
#             {%scripts%}
#             {%renderer%}
#         </footer>
#     </body>
# </html>
# '''
#
# # Page layout
# app.layout = html.Div([
#     header,
#     html.Div(
#         dash.page_container,
#         style={
#             'marginTop': '7vh',  # Creates space exactly matching the header height
#             'padding': '0',  # No padding around the content
#             'flex': '1',  # Allow the content to take the remaining space
#         }
#     )
# ], style={
#     'display': 'flex',
#     'flexDirection': 'column',
#     'margin': 0,
#     'padding': 0,
#     'minHeight': '100vh',  # Full height layout
# })
#
# ======================  main.py  ====================================
import time
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler

from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

from data_loader.FI2010 import Dataset_fi2010
from models.gttlob import (
    snapshot_to_graph,
    GraphTemporalTransformer,
    collate_graph_sequences,        # déjà défini dans models.gttlob
)

# ------------------- hyper-paramètres « rapides » --------------------
T_WINDOW       = 50          # longueur fenêtre
LIGHTEN        = True        # 20 lignes LOB
STRIDE         = 20          # saute 19 observations
BATCH_TRAIN    = 64
BATCH_EVAL     = 128
NUM_WORKERS    = 8
EPOCHS         = 20
PATIENCE       = 4           # early-stopping

CKPT_DIR  = Path(__file__).parent / "weights"
CKPT_DIR.mkdir(exist_ok=True, parents=True)
CKPT_FILE = CKPT_DIR / "gttlob_fast.ckpt"

device = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------------------------------------------------------


# ---------------------- Dataset fenêtre → graphes --------------------
class GraphWindowDataset(Dataset):
    """Convertit les fenêtres FI-2010 en liste de graphes PyG."""
    def __init__(self, fi_ds, *, stride=1, local_norm=True):
        self.fi_ds      = fi_ds
        self.stride     = stride
        self.local_norm = local_norm

    def __len__(self):
        return (len(self.fi_ds) - 1) // self.stride + 1

    def __getitem__(self, idx):
        window, label = self.fi_ds[idx * self.stride]   # (1,T,40)
        window = window.squeeze(0).numpy()              # (T,40)

        ref = window[0, 2] if self.local_norm else None
        graphs = [snapshot_to_graph(snap, first_price_ref=ref)
                  for snap in window]
        return graphs, label.to(torch.long)


# --------------------------- DataLoader ------------------------------
def make_loader(days, *, shuffle, batch):
    raw_ds = Dataset_fi2010(
        auction=False, normalization="Zscore",
        stock_idx=[0, 1, 2, 3, 4],
        days=days, T=T_WINDOW, k=4, lighten=LIGHTEN
    )
    ds = GraphWindowDataset(raw_ds, stride=STRIDE)
    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=shuffle,
        collate_fn=collate_graph_sequences,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        pin_memory=True
    )


# ------------------------- Boucle d'entraînement ---------------------
scaler = GradScaler()

def train_epoch(model, loader, opt, sched, log_every=25):
    model.train()
    tot_loss, t0 = 0.0, time.perf_counter()

    for i, (seqs, y) in tqdm(enumerate(loader, 1)):
        seqs = [[g.to(device) for g in s] for s in seqs]
        y    = y.to(device)

        with autocast():
            out  = model(seqs)
            loss = model.compute_loss(out, y)

        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        sched.step()

        tot_loss += loss.item() * len(y)

        # ---- log débit toutes les `log_every` itérations ----
        if i % log_every == 0:
            ips = log_every / (time.perf_counter() - t0)
            eta = (len(loader) - i) / ips / 60
            print(f"[{i:>5}/{len(loader)}] {ips:6.2f} it/s  ETA {eta:4.1f} min")
            t0 = time.perf_counter()

    return tot_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader):
    model.eval(); pred, true = [], []
    for seqs, y in loader:
        seqs = [[g.to(device) for g in s] for s in seqs]
        logits = model(seqs)["logits"]
        pred.extend(logits.argmax(1).cpu())
        true.extend(y)
    acc = accuracy_score(true, pred)
    f1  = f1_score(true, pred, average="macro")
    mcc = matthews_corrcoef(true, pred)
    return acc, f1, mcc


# ----------------------------- Main ----------------------------------
def main():
    train_ld = make_loader([1, 2, 3, 4, 5, 6, 7], shuffle=True,  batch=BATCH_TRAIN)
    val_ld   = make_loader([8, 9],             shuffle=False, batch=BATCH_EVAL)

    model = GraphTemporalTransformer().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=3e-4, steps_per_epoch=len(train_ld), epochs=EPOCHS
    )

    best_f1, wait, best_state = 0.0, 0, None
    for epoch in tqdm(range(EPOCHS)):
        loss = train_epoch(model, train_ld, opt, sched)
        acc, f1, mcc = eval_epoch(model, val_ld)
        print(f"E{epoch:02d}  loss {loss:.4f}  valF1 {f1:.3f}")

        if f1 > best_f1:
            best_f1, best_state, wait = f1, model.state_dict(), 0
        else:
            wait += 1
            if wait == PATIENCE:
                print("Early stop")
                break

    torch.save(best_state, CKPT_FILE)
    print(f"Best val F1 = {best_f1:.3f}  •  checkpoint enregistré → {CKPT_FILE}")

    # --------- test final sur Day 10 ----------
    test_ld = make_loader([10], shuffle=False, batch=BATCH_EVAL)
    model.load_state_dict(torch.load(CKPT_FILE, map_location=device))
    acc, f1, mcc = eval_epoch(model, test_ld)
    print(f"TEST  Acc {acc:.3f}  F1 {f1:.3f}  MCC {mcc:.3f}")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
# =====================================================================
