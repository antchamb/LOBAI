from data_loader.FI2010 import *
import matplotlib.pyplot as plt

# stocks = [0, 1, 2]
days = [1, 2]
norm="Zscore"
# for stock in stocks:
#     print(f"Stock {stock} data:")
#     ds = Dataset_fi2010(
#         auction=False,
#         normalization=norm,
#         stock_idx=[stock],
#         days=days,
#         T=1,
#         k=0,
#         lighten=True,
#     )
#     print(ds.get_spread())
ds = Dataset_fi2010(
        auction=False,
        normalization=norm,
        stock_idx=[0],
        days=days,
        T=1,
        k=0,
        lighten=True,
)
spread_data_one_day = ds.get_spread()
tick_time = np.arange(len(spread_data_one_day))

spdata = Dataset_fi2010(
    auction=False,
    normalization=norm,
    stock_idx=[0],
    days=[1],
    T=1,
    k=0,
    lighten=True,
).get_spread()
tickt = np.arange(len(spdata))
plt.plot(tick_time, spread_data_one_day, label='Stock 0')
plt.show()
plt.plot(tickt, spdata, label='Stock 0 Day 1')
