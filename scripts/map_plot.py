import numpy as np
from pathlib import Path
from datetime import time
from plot import plot_map


def read_data(files, period) -> tuple:
    for file in files:
        if "MLS" in file.name:
            mls = np.load(file, allow_pickle=True).item()
        else:
            mira2 = np.load(file, allow_pickle=True).item()
    return get_period(mls, period=period), get_period(mira2, period=period)


def get_files() -> list:
    cwd = Path(__file__).parent
    datadir = cwd / "data"
    return [file for file in datadir.glob(pattern="*.npy")]


def get_period(data, period="day") -> dict:
    match period:
        case "day":
            lower = time(hour=10, minute=0, second=0)
            upper = time(hour=14, minute=0, second=0)
        case "night":
            lower = time(hour=0, minute=0, second=0)
            upper = time(hour=4, minute=0, second=0)

    dct = {}

    for dt, val in data.items():
        t = dt.time()
        if lower <= t <= upper:
            dct[dt] = val
    return dct


def reduce_dict(data):
    rd = {}

    for dt, val in data.items():
        lat = val["lat"]
        lon = val["lon"]

        if lat >= 66.5 and 19 <= lon <= 22.8:
            rd[dt] = val

    return rd


def save_mls(data) -> None:
    home = Path(__file__).home()
    downloads = home / "Downloads"
    np.save(downloads / "MLS.O3.reduced.npy", data, allow_pickle=True)


files = get_files()
mls, mira2 = read_data(files=files, period="day")
plot_map(data=reduce_dict(mls), period="day")
save_mls(data=reduce_dict(mls))
