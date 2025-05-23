# import torch
#
# ckpt = torch.load("LOBAPP/weights/deeplob_light.pt", map_location="cpu")
#
# print("keys :", list(ckpt)[:5], "...")
# print("num tensors :", len(ckpt))
# print("first tensor shape :", ckpt['conv1.0.weight'].shape)
# LOBAPP/test.py   ← you are here
from pathlib import Path
import torch, pprint

# 1️⃣  Point to ../weights/deeplob_light.pt
WEIGHT_DIR = Path(__file__).with_name("weights")        # LOBAPP/weights
ckpt_path  = WEIGHT_DIR / "deeplob_light.pt"            # full filename

print("Expecting file :", ckpt_path)
assert ckpt_path.exists(), "❌ checkpoint not found, wrong name?"

# 2️⃣  Safe load (weights_only avoids the pickle warning)
state = torch.load(ckpt_path, map_location="cpu", weights_only=True)

# 3️⃣  Inspect a few tensors
print("first keys :", list(state)[:5], "…")
print("num tensors:", len(state))
print("conv1.0.weight shape :", state["conv1.0.weight"].shape)
