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
from models.gttlob import *
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from data_loader.FI2010 import Dataset_fi2010
# + toutes tes classes GNN/Transformer déjà définies

CKPT_DIR  = Path(__file__).parent / "weights"
CKPT_DIR.mkdir(exist_ok=True)
CKPT_FILE = CKPT_DIR / "gttlob_k10_7days.pt"

def make_loader(days, shuffle, bs):
    ds = GraphWindowDataset(..., stride=20)
    return DataLoader(ds,
                      batch_size=bs,
                      shuffle=shuffle,
                      collate_fn=collate_graph_sequences,
                      num_workers=8,           # ← au lieu de 0
                      persistent_workers=True, # ← reste chargés
                      pin_memory=True)


from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
import time

# def train_epoch(model, loader, opt, sched):
#     model.train(); tot = 0
#     for seqs, y in loader:
#         seqs = [[g.to(device) for g in s] for s in seqs]
#         y    = y.to(device)
#
#         with autocast():                         # ← nouveau
#             out  = model(seqs)
#             loss = model.compute_loss(out, y)
#
#         opt.zero_grad()
#         scaler.scale(loss).backward()            # FP16 safe
#         scaler.unscale_(opt)
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         scaler.step(opt)
#         scaler.update()
#         sched.step()
#
#         tot += loss.item()*len(y)
#     return tot/len(loader.dataset)
from torch.cuda.amp import autocast, GradScaler
import time

scaler = GradScaler()

def train_epoch(model, loader, opt, sched, log_every=50):
    model.train()
    tot_loss = 0

    start = time.perf_counter()         # ← chrono
    for batch_idx, (seqs, y) in enumerate(loader, 1):
        # ----------- mise sur GPU -----------
        seqs = [[g.to(device) for g in s] for s in seqs]
        y    = y.to(device)

        # ----------- forward FP16 -----------
        with autocast():
            out  = model(seqs)
            loss = model.compute_loss(out, y)

        # ----------- backward FP16 ----------
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        # ----------- stats ------------------
        tot_loss += loss.item() * len(y)

        # ----- affichage débit 1× toutes `log_every` itérations -----
        if batch_idx % log_every == 0:
            it_per_s = log_every / (time.perf_counter() - start)
            eta_min  = (len(loader) - batch_idx) / it_per_s / 60
            print(f"[{batch_idx:>5}/{len(loader)}] "
                  f"{it_per_s:5.2f} it/s   ETA ≃ {eta_min:4.1f} min")
            start = time.perf_counter()  # remet le chrono à zéro

    return tot_loss / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader):
    model.eval(); p,t = [],[]
    for seqs,y in loader:
        seqs = [[g.to(device) for g in s] for s in seqs]
        logits = model(seqs)['logits']
        p.extend(logits.argmax(1).cpu()); t.extend(y)
    acc = accuracy_score(t,p); f1 = f1_score(t,p,average='macro')
    mcc = matthews_corrcoef(t,p)
    return acc,f1,mcc

def main():
    train_ld = make_loader([1,2,3,4,5,6,7],True,64)
    val_ld   = make_loader([8,9],False,64)
    model = GraphTemporalTransformer().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4,
                                                steps_per_epoch=len(train_ld),
                                                epochs=20)
    best_f1, patience = 0, 4
    wait = 0
    for ep in tqdm(range(100)):
        loss = train_epoch(model, train_ld, opt, sched)
        acc,f1,mcc = eval_epoch(model, val_ld)
        print(f"E{ep}  loss {loss:.4f}  valF1 {f1:.3f}")
        if f1>best_f1: best_f1,f,best_state = f1,acc,model.state_dict(); wait=0
        else: wait+=1
        if wait==patience: break
    CKPT_FILE.write_bytes(torch.save(best_state, CKPT_FILE))

    # test on day 10
    test_ld = make_loader([10],False,64)
    model.load_state_dict(torch.load(CKPT_FILE))
    acc,f1,mcc = eval_epoch(model,test_ld)
    print(f"TEST  acc {acc:.3f}  f1 {f1:.3f}  mcc {mcc:.3f}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    main()
