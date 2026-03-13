import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from ozone.io import get_datadir
from ozone.analysis import read_npy
import pandas as pd
from datetime import datetime


def find_files(dataset):
    cwd = Path(__file__).parent
    datadir = cwd / "data"

    match dataset:
        case "mira2":
            d = datadir / "matching"
            data = d / "mira2_matching.npy"
        case "mls":
            d = datadir / "smooth"
            data = d / "mls_smooth.npy"
        case "pgrid":
            d = datadir / "matching"
            data = d / "pgrid.npy"

    return data


daterange = read_npy(file=get_datadir() / "daterange.npy")
mls = read_npy(find_files(dataset="mls"))
mira2 = read_npy(find_files(dataset="mira2"))
pgrid = read_npy(find_files(dataset="pgrid"))


# make mls
mls_date = np.array([dt.date() for dt in mls.keys()])
mlsO3 = {dt: val["O3smooth"] for dt, val in mls.items()}

for i, d in enumerate(daterange):
    if d not in mls_date:
        dt = datetime(
            year=d.year, month=d.month, day=d.day, hour=12, minute=0, second=0
        )
        mlsO3[dt] = np.full(shape=41, fill_value=np.nan)

dt_sorted = sorted(mlsO3.keys())
mlsO3_sorted = {}
for dt in dt_sorted:
    mlsO3_sorted[dt] = mlsO3[dt]


# make mira2
m2_date = np.array([dt.date() for dt in mira2.keys()])
m2O3 = {dt: val["apriori"] * val["x"] for dt, val in mira2.items()}

for i, d in enumerate(daterange):
    if d not in m2_date:
        dt = datetime(
            year=d.year, month=d.month, day=d.day, hour=12, minute=0, second=0
        )
        m2O3[dt] = np.full(shape=41, fill_value=np.nan)
