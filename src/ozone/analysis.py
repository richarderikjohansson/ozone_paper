import numpy as np
from datetime import datetime
from datetime import time
from datetime import timedelta
from scipy.interpolate import interp1d
from numpy.typing import NDArray

from .io import get_downloadsdir


class MatchData:
    def __init__(self, mira2, mls, logger):
        self.logger = logger
        self.mira2_file = mira2
        self.mls_file = mls
        self.mira2 = np.load(mira2, allow_pickle=True).item()
        self.mls = np.load(mls, allow_pickle=True).item()
        self.match_mira2_and_mls()
        self.interp_mls()
        self.save_matching_data()

    def match_mira2_and_mls(self):
        tup = match_measurements(self.mira2, self.mls)
        self.avk_match = tup[0]
        self.mls_match = tup[1]
        self.mira2_match = tup[2]

    def interp_mls(self):
        mira2pres = np.array([v["pgrid"] for v in self.mira2_match.values()])
        mira2apriori = np.array([v["apriori"] for v in self.mira2_match.values()])

        ptarget = mira2pres[0]
        apriori = mira2apriori[0]
        interp_mls(self.avk_match, self.mls_match, ptarget, apriori)

    def save_matching_data(self):
        outdir = get_downloadsdir()
        mira2_fn = self.mira2_file.stem + "_matching.npy"
        mls_fn = self.mls_file.stem + "_matching.npy"

        np.save(outdir / mira2_fn, self.mira2_match, allow_pickle=True)
        np.save(outdir / mls_fn, self.mls_match, allow_pickle=True)
        self.logger.info(f"Saved matching MIRA2 data in {outdir / mira2_fn}")
        self.logger.info(f"Saved matching MLS data in {outdir / mls_fn}")


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


def match_measurements(mira2, mls):
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
        psource = mls[dt]["pressure"] * 1e2
        vmr = mls[dt]["O3"]
        precision = mls[dt]["precision"]

        mls[dt]["p_interp"] = ptarget
        log_src = np.log(psource)
        log_tgt = np.log(ptarget)
        interp_vmr = interp1d(
            log_src,
            vmr,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        interp_precision = interp1d(
            log_src,
            precision,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        mls[dt]["O3_interp"] = interp_vmr(log_tgt)
        mls[dt]["O3_interp_smooth"] = smooth_mls(
            avk=avk[dt],
            mls=mls[dt],
            apriori=apriori,
        )
        mls[dt]["precision_interp"] = interp_precision(log_tgt)

    return mls


def smooth_mls(avk, mls, apriori):
    shape = avk.shape[0]
    prod = mls["O3_interp"]
    identity = np.identity(n=shape)
    smoothed = (identity - avk) @ apriori + avk @ prod
    return smoothed


def read_npy(file):
    try:
        data = np.load(file, allow_pickle=True).item()
    except ValueError:
        data = np.load(file, allow_pickle=True)

    return data


def pressure_thickness(p):
    logp = np.log(p)
    logp_edges = np.empty(len(p) + 1)

    logp_edges[1:-1] = 0.5 * (logp[:-1] + logp[1:])
    logp_edges[0] = logp[0] + 0.5 * (logp[0] - logp[1])
    logp_edges[-1] = logp[-1] + 0.5 * (logp[-1] - logp[-2])

    p_edges = np.exp(logp_edges)
    dp = np.abs(np.diff(p_edges))

    return dp


def make_weighted_mean(vmr: NDArray, p: NDArray, s: int, e: int) -> NDArray:
    dp = pressure_thickness(p)
    w = np.zeros(len(vmr))
    w[s:e] = dp[s:e] / np.sum(dp[s:e])
    vmr_mean = w @ vmr
    return vmr_mean


def propagate_uncertainty_mls(
    precision: NDArray,
    p: NDArray,
    s: int,
    e: int,
) -> float:
    dp = pressure_thickness(p)
    w = np.zeros(len(p))
    w[s:e] = dp[s:e] / np.sum(dp[s:e])

    var = np.array([sigma * sigma for sigma in precision])
    varx = w @ var
    sigma = np.sqrt(varx)
    return sigma


def propagate_uncertainty_mira2(
    covar: NDArray[NDArray, NDArray],
    p: NDArray,
    s: int,
    e: int,
) -> float:
    dp = pressure_thickness(p)
    w = np.zeros(len(p))
    w[s:e] = dp[s:e] / np.sum(dp[s:e])
    varx = w @ covar @ w
    sigma = np.sqrt(varx)
    return sigma
