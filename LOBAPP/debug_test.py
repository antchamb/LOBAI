from data_loader.FI2010 import *
import torch

ofi_ds = OFIDataset(
    auction=False, normalization="Zscore",
    stock_idx=[0,1,2,3,4],
    days=[1],
    T=100,    # look-back
    k=2,      # prediction horizon
    lighten=False,
)
loader = torch.utils.data.DataLoader(ofi_ds, batch_size=64, shuffle=True)

ds = Dataset_fi2010(
    auction=False, normalization="Zscore",
    stock_idx=[0,1,2,3,4],
    days=[1],
    T=100,    # look-back
    k=2,      # prediction horizon
    lighten=False,
)
load = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)
