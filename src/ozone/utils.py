from pathlib import Path
import numpy as np
from .io import exportdir
from datetime import datetime


def find_downloads() -> Path:
    """Function to locate and return Downloads directory

    This function locates the Downloads directory and is
    assumed to be located in $HOME/Downloads. If this
    directory do not exist will it be created

    Returns:
        [TODO:description]
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

    daterange = np.load(exportdir() / "daterange.npy", allow_pickle=True)

    for d in daterange:
        if d not in keys:
            dtfill = datetime(year=d.year,
                              month=d.month,
                              day=d.day,
                              hour=12,
                              minute=0,
                              second=0)
            mdict[dtfill] = {k: np.full(s, np.nan) for k, s in shapes.items()}

    dict_keys_sorted = sorted(mdict.keys())
    sdict = {k: mdict[k] for k in dict_keys_sorted}
    return sdict
