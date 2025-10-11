import numpy as np
from datetime import datetime
from datetime import time
from datetime import timedelta
from .io import get_screendir, get_datadir
from .utils import fill_nan


def get_period(data: dict, period: str):
    """Function to get data from period

    This function extracts data from MIRA2 or MLS
    =- 2 hours from either 12:00 AM or 02:00 AM

    Args:
        data: MIRA2 or MLS data dictionary
        period: string to decide either day or night
    """

    rdata = {}
    match period:
        case "day":
            t0 = time(hour=10, minute=0, second=0)
            t1 = time(hour=14, minute=0, second=0)

        case "night":
            t0 = time(hour=0, minute=0, second=0)
            t1 = time(hour=4, minute=0, second=0)

    for dt, vals in data.items():
        if t0 <= dt.time() <= t1:
            rdata[dt] = vals
    return rdata


def mk_daterange(period: str):
    """Function to make a daterange

    This function creates a daterange either for day or night.
    day range is considered to start on 191001 - 200430 at 12:00 AM.
    Night range will be the same dates but for 02:00 AM


    Args:
        period: string to decide either day or night
    """
    match period:
        case "day":
            hour = 12
        case "night":
            hour = 2

    start = datetime(
        year=2019,
        month=10,
        day=1,
        hour=hour,
        minute=0,
        second=0,
    )
    end = datetime(
        year=2020,
        month=4,
        day=30,
        hour=hour,
        minute=0,
        second=0,
    )
    daterange = []
    current = start

    while current <= end:
        daterange.append(current)
        current += timedelta(days=1)

    return np.array(daterange)


class MLSRegridding:
    def __init__(self, dataset, period, logger):
        self.dataset = dataset
        self.period = period
        self.logger = logger
        self.daterange = mk_daterange(period=self.period)

    def read_data(self):
        screendir = get_screendir()
        filepath = screendir / f"{self.dataset}.npy"

        if filepath.exists():
            self.logger.info(f"Reading MLS {self.dataset}")
            data = np.load(filepath, allow_pickle=True).item()
            drange = mk_daterange(period=self.period)
            temp = get_period(data=data, period=self.period)
            self.data = fill_nan(data=temp, drange=drange)
        else:
            self.logger.info(f"Could not find {self.dataset}")

    def regrid_MLS(self):
        psource = np.load(get_datadir() / "pgrid.npy", allow_pickle=True)
