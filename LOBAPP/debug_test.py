# import numpy as np
#
# from data_loader.FI2010 import Dataset_fi2010
#
# stocks= [0]
#
# days = [1,2]
# norm="Zscore"
# graphs = []
#
# ds = Dataset_fi2010(
#     auction=False,
#     normalization=norm,
#     stock_idx=stocks,
#     days=days,
#     T=1,
#     k=0,  # k is not used in OFI calculation
#     lighten=True,
# )
#
# a, b, c =ds.get_ofi()

# tick_time = np.arange(len(ofi_series))