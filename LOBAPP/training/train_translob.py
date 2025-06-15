from pathlib import Path
import torch

from training.train_deeplob import DATA_ROOT, BATCH_SIZE

print("CUDA visible :", torch.cuda.is_available())          # → True
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

train_set = FI2010D(
    auction=False,
    normalization="Zscore",
    stock_idx=[0, 1, 2, 3, 4],   # all five stocks
    days=[1,2,3,4,5,6,7], # day 1 = training set; 2-10 = test sets
    T=T_WINDOW,
    k=K_HORIZON,
    lighten=LIGHTEN,
)