import numpy as np
from datetime import datetime
from datetime import time
from datetime import timedelta
from .io import get_screendir, get_datadir
from .utils import fill_nan, find_downloads
from scipy.interpolate import interp1d


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
    """
    Class to regrid MLS data to MIRA2 pressure grid
    """

    def __init__(self, dataset, period, logger):
        self.dataset = dataset
        self.period = period
        self.logger = logger
        self.daterange = mk_daterange(period=self.period)

    def read_data(self):
        """Method to read screened MLS data from ~/.cache/screen
        """
        screendir = get_screendir()
        filepath = screendir / f"{self.dataset}.npy"

        if filepath.exists():
            self.found = True
            data = np.load(filepath, allow_pickle=True).item()
            drange = mk_daterange(period=self.period)
            temp = get_period(data=data, period=self.period)
            self.data = fill_nan(data=temp, drange=drange)
        else:
            self.found = False
            self.logger.warning(f"Could not find {self.dataset}")

    def regrid_MLS(self):
        """Method to grid MLS data to MIRA2 pressure grid

        Reads mean pressure file from MIRA2. Then interpolates
        the MLS data to that pressure grid
        """
        p_target = np.load(get_datadir() / "pgrid.npy", allow_pickle=True)

        for v in self.data.values():
            if not np.isnan(v[self.dataset]).any():
                p_source = v["pressure"]
                product = v[self.dataset]
                logp_src = np.log(p_source)
                logp_tgt = np.log(p_target)

                # check if another interpolation kind is better suited
                interp = interp1d(
                    logp_src,
                    product,
                    kind="linear",
                    bounds_error=False,
                    fill_value=np.nan,
                )
                productnew = interp(logp_tgt)
                v["p_coarse"] = p_target
                v[f"{self.dataset}_coarse"] = productnew
            else:
                v["p_coarse"] = np.full_like(p_target, np.nan),
                v[f"{self.dataset}_coarse"] = np.full_like(p_target, np.nan)

        fn = f"{self.dataset}{self.period}_regridded.npy"
        savepath = find_downloads() / fn
        self.logger.info(f"Saving regridded {self.dataset} in {savepath}")
        np.save(savepath, self.data, allow_pickle=True)
