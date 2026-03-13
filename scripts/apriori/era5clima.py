import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import numpy as np
from datetime import datetime
from scipy.signal import savgol_filter

MM_2_VMR = 28.9644 / 47.9982


def get_all_files():
    cwd = Path(__file__).parent
    ddir = cwd / "era5"
    generator = ddir.rglob("*.nc")
    return np.array([file for file in generator])


def generate_monthly_means(file):
    ds = xr.open_dataset(file)
    time = ds["o3"]["valid_time"].to_numpy()
    products = ds["o3"][:, :, 0, 0].to_numpy()
    timestamps = make_datetime(time)

    dct = {
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
        6: [],
        7: [],
        8: [],
        9: [],
        10: [],
        11: [],
        12: [],
    }
    for dt, profile in zip(timestamps, products):
        month = dt.month
        smoothed = savgol_filter(profile, window_length=30, polyorder=5)
        match month:
            case 1:
                dct[1].append(smoothed)
            case 2:
                dct[2].append(smoothed)
            case 3:
                dct[3].append(smoothed)
            case 4:
                dct[4].append(smoothed)
            case 5:
                dct[5].append(smoothed)
            case 6:
                dct[6].append(smoothed)
            case 7:
                dct[7].append(smoothed)
            case 8:
                dct[8].append(smoothed)
            case 9:
                dct[9].append(smoothed)
            case 10:
                dct[10].append(smoothed)
            case 11:
                dct[11].append(smoothed)
            case 12:
                dct[12].append(smoothed)

    means = {k: np.mean(v, axis=0) * MM_2_VMR for k, v in dct.items()}
    fn = "monthly_means/" + file.stem + "_monthly_means.npy"
    np.save(fn, means, allow_pickle=True)


def make_datetime(time):
    dts = []
    for t in time:
        dtstr = str(t).split(".")[:-1][0]
        dt = datetime.strptime(dtstr, "%Y-%m-%dT%H:%M:%S")
        dts.append(dt)

    return np.array(dts)


files = get_all_files()
z = np.loadtxt("era5/z.txt")

for file in files:
    generate_monthly_means(file)
