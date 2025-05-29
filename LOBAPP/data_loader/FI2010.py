import os
import numpy as np
import torch
from lean.models import file_path
from tqdm import tqdm
from matplotlib.rcsetup import validate_backend


def __get_raw__(auction, normalization, day):
    """
    Handling function for loading raw FI2010 dataset
    Parameters
    ----------
    auction: {True, False}
    normalization: {'Zscore', 'MinMax', 'DecPre'}
    day: {1, 2, ..., 10}
    """

    root_path = r"D:\Bureau\FI-2010"

    if auction:
        path1 = "Auction"
    else:
        path1 = "NoAuction"

    if normalization == 'Zscore':
        tmp_path_1 = '1.'
    elif normalization == 'MinMax':
        tmp_path_1 = '2.'
    elif normalization == 'DecPre':
        tmp_path_1 = '3.'
    else:
        raise ValueError('Invalid Normalization type or day')

    tmp_path_2 = f"{path1}_{normalization}"
    path2 = f"{tmp_path_1}{tmp_path_2}"

    if day == 1:
        path3 = tmp_path_2 + '_Training'
        filename = f"Train_Dst_{path1}_{normalization}_CF_{str(day)}.txt"
    else:
        path3 = tmp_path_2 + '_Testing'
        day = day - 1

        filename = f"Test_Dst_{path1}_{normalization}_CF_{str(day)}.txt"


    file_path = os.path.join(root_path, path1, path2, path3, filename)

    try:
        fi2010_dataset = np.loadtxt(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading the file: {e}")

    return fi2010_dataset



def __extract_stock__(raw_data, stock_idx):
    """
    Extract specific stock data from raw FI2010 dataset
    Parameters
    ----------
    raw_data: Numpy Array
    stock_idx: {0, 1, 2, 3, 4}
    """
    n_boundaries = 4
    boundaries = np.sort(
        np.argsort(np.abs(np.diff(raw_data[0], prepend=np.inf)))[-n_boundaries - 1:]
    )
    boundaries = np.append(boundaries, [raw_data.shape[1]])
    split_data = tuple(raw_data[:, boundaries[i] : boundaries[i + 1]] for i in range(n_boundaries + 1))
    return split_data[stock_idx]


def __split_x_y__(data, lighten, rows=None):
    """
    Extract lob data and annotated label from fi-2010 data
    Parameters
    ----------
    data: Numpy Array
    """
    if lighten:
        data_length = 20
    else:
        data_length = 144

    if rows is not None:
        data_length = rows

    x = data[:data_length, :].T
    y = data[-5:, :].T
    return x, y



def __data_processing__(x, y, T, k):
    """
    Process whole time-series-data
    Parameters
    ----------
    x: Numpy Array of LOB
    y: Numpy Array of annotated label
    T: Length of time frame in single input data
    k: Prediction horizon{0, 1, 2, 3, 4}
    """
    [N, D] = x.shape

    # x processing
    x_proc = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        x_proc[i - T] = x[i - T:i, :]

    # y processing
    y_proc = y[T - 1:N]
    y_proc = y_proc[:, k] - 1
    return x_proc, y_proc


class Dataset_fi2010:
    def __init__(self, auction, normalization, stock_idx, days, T, k, lighten):
        """ Initialization """
        self.auction = auction
        self.normalization = normalization
        self.days = days
        self.stock_idx = stock_idx
        self.T = T
        self.k = k
        self.lighten = lighten

        x, y = self.__init_dataset__()
        x = torch.from_numpy(x)
        self.x = torch.unsqueeze(x, 1)
        self.y = torch.from_numpy(y)

        self.length = len(y)
        self.labels_count = self.get_label()

    def __init_dataset__(self):
        x_cat = np.array([])
        y_cat = np.array([])
        for stock in self.stock_idx:
            for day in self.days:
                day_data = __extract_stock__(
                    __get_raw__(auction=self.auction, normalization=self.normalization, day=day), stock)
                x, y = __split_x_y__(day_data, self.lighten)
                x_day, y_day = __data_processing__(x, y, self.T, self.k)

                if len(x_cat) == 0 and len(y_cat) == 0:
                    x_cat = x_day
                    y_cat = y_day
                else:
                    x_cat = np.concatenate((x_cat, x_day), axis=0)
                    y_cat = np.concatenate((y_cat, y_day), axis=0)

        x_cat = x_cat.astype(np.float32)
        y_cat = y_cat.astype(np.int64)
        return x_cat, y_cat

    def __len__(self):
        """Denotes the total number of samples"""
        return self.length

    def __getitem__(self, index):
        """Generates samples of data"""
        return self.x[index], self.y[index]

    def _get_midprice(self) -> np.ndarray:
        """
        Returns a 1-D NumPy array of mid-prices (Ask₁ + Bid₁)/2
        for every snapshot in the dataset.
        Works both for lighten=True (20 rows) and full (40/144 rows).
        """
        ask_row = 0  # first row in each 4-row block
        bid_row = 2  # third row in each 4-row block
        mid = 0.5 * (
                self.x[:, 0, :, ask_row].squeeze() +
                self.x[:, 0, :, bid_row].squeeze()
        )
        return mid.astype(np.float64)

    def get_label(self):
        """Returns label counts as a dictionary: {'down': count, 'flat': count, 'up': count}"""

        label_map = {0: 'down', 1: 'flat', 2: 'up'}
        y_np = self.y.cpu().numpy() if isinstance(self.y, torch.Tensor) else self.y
        unique, counts = np.unique(y_np.astype(int), return_counts=True)

        return {label_map[k]: v for k, v in zip(unique, counts)}

    @staticmethod
    def handling_level(level: int, T, lighten: bool = True):
        if T != 1:
            raise ValueError("`get_spread` requires T=1 for snapshot-level computation.")

        if level < 1:
            raise ValueError("Level must be at least 1.")

        rows_per_level = 4
        base = (level - 1) * rows_per_level
        ask_row = base
        bid_row = base + 2
        max_level = 5 if lighten else 10
        if level > max_level:
            raise ValueError(f"Level must be at most {max_level}.")

        return rows_per_level, base, ask_row, bid_row, max_level

    def get_spread(self, level: int = 1, agg: str | None = "series"):
        """
            Computes the spread between the ask and bid prices for a stock at each update
            Parameters
            ----------
            level : int
                Order-book level (1 = top-of-book).
                Valid range: 1-5 when `lighten=True`, 1-10 otherwise.
            agg   : {"series", "mean", "median"} or None
                * "series"  → numpy array of spreads (default)
                * "mean"    → scalar mean spread
                * "median"  → scalar median spread

            Notes
            -----
            Even though the dataset is normalised (Z-score / Min-Max / DecPre),
            the **difference** `ask − bid` still reflects the true shape of the
            spread, because both prices are transformed with the *same* linear
            mapping.
        """

        rows_per_level, base, ask_row, bid_row, max_level = self.handling_level(level, self.T, self.lighten)

        # Compute spread for each snapshot
        spreads = (self.x[:, 0, :, ask_row] - self.x[:, 0, :, bid_row]).squeeze()

        if agg == "series":
            return np.array(spreads)
        elif agg == "mean":
            return np.array(spreads.mean())
        elif agg == "median":
            return np.array(np.median(spreads, axis=0))
        else:
            raise ValueError("Invalid aggregation method. Use 'series', 'mean', or 'median'.")

    def get_ofi(self, level: int = 1, agg: str | None = "series"):
        """
        Computes the order flow imbalance (OFI) for a stock at each update.
        Parameters
        ----------
        level : int
            Order-book level (1 = top-of-book).
            Valid range: 1-5 when `lighten=True`, 1-10 otherwise.
        agg   : {"series", "mean", "median"} or None
            * "series"  → numpy array of OFI values (default)
            * "mean"    → scalar mean OFI
            * "median"  → scalar median OFI

        Notes
        -----
        The OFI is computed as the difference between the total volume on the ask side and the total volume on the bid side.
        """

        rows_per_level, base, ask_row, bid_row, max_level = self.handling_level(level, self.T, self.lighten)
        print(self.x.squeeze().shape)

        snap = self.x[:, 0, 0, :].cpu().numpy()

        n = snap.shape[0]
        ofi = np.zeros(n, dtype=np.float64)

        lvl_list = list(range(1, max_level + 1))

        for lvl in tqdm(lvl_list):
            base = 4 * (lvl - 1)
            pa = snap[:, base]
            va = snap[:, base + 1]
            pb = snap[:, base + 2]
            vb = snap[:, base + 3]

            # bid order flow
            bOF = np.zeros_like(pb)
            diff_b = pb[1:] - pb[:-1]
            mask_up = diff_b > 0
            mask_same = diff_b == 0
            mask_down = diff_b < 0
            bOF[1:][mask_up] = vb[1:][mask_up]
            bOF[1:][mask_same] = vb[1:][mask_same] - vb[:-1][mask_same]
            bOF[1:][mask_down] = -vb[1:][mask_down]

            # ask order flow
            aOF = np.zeros_like(pa)
            diff_a = pa[1:] - pa[:-1]
            mask_up = diff_a > 0
            mask_same = diff_a == 0
            mask_down = diff_a < 0
            aOF[1:][mask_up] = -va[:-1][mask_up]
            aOF[1:][mask_same] = va[1:][mask_same] - va[:-1][mask_same]
            aOF[1:][mask_down] = va[1:][mask_down]

        ofi += bOF + aOF

        return ofi

    def as_2d(self):
        """
        Returns the dataset as a 2D numpy array.
        The first dimension is the sample index, and the second dimension is the feature index.
        """
        return self.x[:, 0, -1, :].cpu().numpy()

    def get_volatility(self, horizon: int = 1, mode: str = "series"):
        """
        μ-free realised volatility of mid-price changes over `horizon`.

        Parameters
        ----------
        horizon : int
            Number of snapshots ahead. Usually in {1,2,3,5,10}.
        mode : {"series","std","var"}
            * "series": vector of absolute returns |Δpₜ|
            * "std"   : scalar σ = √Var(Δpₜ)
            * "var"   : scalar Var(Δpₜ)

        Notes
        -----
        The dataset is normalised, but subtraction keeps the true
        shape of price moves because the transform is affine.  Hence
        σ(|Δp|) is meaningful.
        """
        if horizon < 1:
            raise ValueError("Horizon must be at least 1.")

        mp = self._get_midprice()

        diff = mp[horizon:] - mp[:-horizon]

        if mode == "series":
            return np.abs(diff)
        elif mode == "std":
            return np.std(np.abs(diff))
        elif mode == "var":
            return np.var(np.abs(diff))
        else:
            raise ValueError("Invalid mode. Use 'series', 'std', or 'var'.")

def __vis_sample_lob__():
    import matplotlib.pyplot as plt

    stock = 0
    k = 100
    normalization = 'Zscore'
    day = 9
    idx = 1000
    lighten = True

    day_data = __extract_stock__(
        __get_raw__(auction=False, normalization=normalization, day=day), stock)
    x, y = __split_x_y__(day_data, lighten)
    sample_shot = np.transpose(x[0 + idx:100 + idx])

    image = np.zeros(sample_shot.shape)
    for i in range(5):
        image[14 - i , :] = sample_shot[4 * i, :]
        image[4 - i, :] = sample_shot[4 * i + 1, :]
        image[15 + i, :] = sample_shot[4 * i + 2, :]
        image[5 + i, :] = sample_shot[4 * i + 3, :]

    plt.imshow(image)
    plt.title('Sample LOB from FI-2010 dataset')
    plt.colorbar()
    plt.show()

fi2010 = Dataset_fi2010(auction=False, normalization="Zscore", stock_idx=[0, 1, 2, 3, 4], days=[1], T = 10, k=2, lighten=True)
