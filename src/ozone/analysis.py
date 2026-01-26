import numpy as np
from datetime import datetime
from datetime import time
from datetime import timedelta
from scipy.interpolate import interp1d


def get_period(data: dict, period: str):
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


def match_measurements(self, mira2, mls):
    m2_dt = [dt for dt in mira2.keys()]
    m2_date = np.array([dt.date() for dt in mira2.keys()])
    mls_dt = np.array([dt for dt in mls.keys()])
    mls_date = np.array([dt.date() for dt in mls.keys()])

    aks = dict()
    mls_matching = dict()
    mira2_matching = dict()

    assert len(mls_dt) == len(mls_date)

    for d, mlsdt in zip(mls_date, mls_dt):
        if d in m2_date:
            idx = np.where(m2_date == d)[0]
            dt = m2_dt[idx[0]]
            aks[mlsdt] = mira2[dt]["avk"]
            mls_matching[mlsdt] = mls[mlsdt]

            for i in idx:
                dt = m2_dt[i]
                mira2_matching[dt] = mira2[dt]

    return aks, mls_matching, mira2_matching


def interp_mls(avk, mls, ptarget, apriori):
    for dt, a in avk.items():
        assert dt in mls.keys()
        psource = mls[dt]["pressure"]
        vmr = mls[dt]["O3"]
        mls[dt]["pnew"] = ptarget

        log_src = np.log(psource)
        log_tgt = np.log(ptarget)
        interp = interp1d(
            log_src,
            vmr,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        mls[dt]["O3new"] = interp(log_tgt)
        mls[dt]["O3smooth"] = smooth_mls(avk=avk[dt], mls=mls[dt], apriori=apriori)

    return mls


def smooth_mls(avk, mls, apriori):
    shape = avk.shape[0]
    prod = mls["O3new"]
    identity = np.identity(n=shape)
    smoothed = (identity - avk) @ apriori + avk @ prod
    return smoothed


def read_npy(file):
    try:
        data = np.load(file, allow_pickle=True).item()
    except ValueError:
        data = np.load(file, allow_pickle=True)

    return data
