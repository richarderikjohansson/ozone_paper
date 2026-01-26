from pathlib import Path
import numpy as np
from datetime import datetime
from dataclasses import dataclass, fields, field
from .io import get_datadir
from datetime import date
from numpy.typing import NDArray


@dataclass
class EdgeData:
    source: str
    doy: NDArray
    date: NDArray
    pv_out: NDArray
    pv_mean: NDArray
    pv_in: NDArray
    pv_stat: NDArray


@dataclass
class MLSData:
    source: str
    dt: NDArray
    prod: NDArray
    pgrid: NDArray
    lon: NDArray
    lat: NDArray
    date: NDArray
    time: NDArray


@dataclass
class MIRA2Data:
    source: str
    dt: NDArray
    prod: NDArray
    err: NDArray
    pgrid: NDArray
    apriori: NDArray
    date: NDArray
    time: NDArray


def find_downloads() -> Path:
    """Function to locate and return Downloads directory

    This function locates the Downloads directory and is
    assumed to be located in $HOME/Downloads. If this
    directory do not exist will it be created

    Returns:
        Path to the downloads directory
    """
    home = Path.home()
    downloadsdir = home / "Downloads"
    if not downloadsdir.exists():
        downloadsdir.mkdir()
    return downloadsdir


def fill_nans(mdict: dict) -> dict:
    """Function to fill NaN's if date is missing

    This function takes the dictionary with data from either
    MLS or MIRA2. If there is dates missing will the data be
    filled with NaN with the correct shape of the data

    Args:
        mdict: Dictionary with data

    Returns:
        Dictionary with missing dates filled with NaN's
    """
    keys = np.array([d.date() for d in mdict.keys()])
    shapes = {}
    for data in mdict.values():
        for key, product in data.items():
            shapes[key] = product.shape
        break

    daterange = np.load(get_datadir() / "daterange.npy", allow_pickle=True)

    for d in daterange:
        if d not in keys:
            dtfill = datetime(
                year=d.year, month=d.month, day=d.day, hour=12, minute=0, second=0
            )
            mdict[dtfill] = {k: np.full(s, np.nan) for k, s in shapes.items()}

    dict_keys_sorted = sorted(mdict.keys())
    sdict = {k: mdict[k] for k in dict_keys_sorted}
    return sdict


def fill_nan(data: dict, drange: np.ndarray) -> dict:
    shapes = {}
    for val in data.values():
        for key, product in val.items():
            shapes[key] = product.shape
        break

    dates = [d.date() for d in data]
    for dt in drange:
        date = dt.date()
        if date not in dates:
            data[dt] = {k: np.full(s, np.nan) for k, s in shapes.items()}

    dict_keys_sorted = sorted(data.keys())
    sdata = {k: data[k] for k in dict_keys_sorted}
    return sdata


def parse_edgefile(file: Path) -> EdgeData:
    """Function to parse edge data file

    :param file: Path to the edgedata file
    :return: EdgeData struct
    """
    doy = []
    dt = []
    pv_out = []
    pv_mean = []
    pv_in = []
    pv_stat = []
    missing = "*****"
    daterange = np.load(get_datadir() / "daterange.npy", allow_pickle=True)

    with open(file, "r") as fh:
        lines = fh.readlines()
        for i, line in enumerate(lines):
            data = line.split()
            if i == 0:
                continue
            d = date.strptime(data[1], "%y%m%d")
            if d not in daterange:
                continue

            for j, field in enumerate(data):
                match j:
                    case 0:
                        f = int(field)
                        doy.append(f)

                    case 1:
                        f = date.strptime(field, "%y%m%d")
                        dt.append(f)

                    case 5:
                        if field == missing:
                            f = np.nan
                        else:
                            f = float(field)
                        pv_out.append(f)

                    case 6:
                        if field == missing:
                            f = np.nan
                        else:
                            f = float(field)
                        pv_mean.append(f)

                    case 7:
                        if field == missing:
                            f = np.nan
                        else:
                            f = float(field)
                        pv_in.append(f)
                    case 8:
                        if field == missing:
                            f = np.nan
                        else:
                            f = float(field)
                        pv_stat.append(f)

    edgedata = EdgeData(
        source=str(file),
        doy=np.array(doy),
        date=np.array(dt),
        pv_out=np.array(pv_out),
        pv_mean=np.array(pv_mean),
        pv_in=np.array(pv_in),
        pv_stat=np.array(pv_stat),
    )

    return edgedata


def filter_edgedata(data: EdgeData, severity: int) -> EdgeData:
    """
    Filter the edgedata based on the severity, The filtering
    compares pv_out for 0, pv_mean for 1 and pv_out for 2
    with pv_stat and if pv_stat is lower that the comparison
    will that data be excluded

    :param data: edgedata dataclass
    :param severity: how aggressive the filtering shall be
    :return: returns the filtered edgedata dataclass
    """
    assert severity in [0, 1, 2]

    stat = data.pv_stat
    match severity:
        case 0:
            comparison = data.pv_out
        case 1:
            comparison = data.pv_mean
        case 2:
            comparison = data.pv_in

    mask = stat >= comparison

    names = [f.name for f in fields(data)]
    for name in names:
        if name != "source":
            arr = getattr(data, name)
            setattr(data, name, arr[mask])

    return data
