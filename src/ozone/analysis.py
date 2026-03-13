import numpy as np
from datetime import datetime
from datetime import time
from datetime import timedelta
from scipy.interpolate import interp1d
from numpy.typing import NDArray
from typing import Dict

from .io import get_downloadsdir, get_egdefiles, get_datadir
from .utils import parse_edgefile, filter_edgedata
from scipy.odr import RealData, Model, ODR
from types import SimpleNamespace


class MatchData:
    def __init__(self, mira2, mls, logger):
        self.logger = logger
        self.mira2_file = mira2
        self.mls_file = mls
        self.mira2 = np.load(mira2, allow_pickle=True).item()
        self.mls = np.load(mls, allow_pickle=True).item()
        self.match_mira2_and_mls()
        if "O3" in self.mls_file.name:
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
    mask = np.isfinite(prod)
    identity = np.identity(n=shape)
    smoothed = (identity[mask][:, mask] - avk[mask][:, mask]) @ apriori[mask] + avk[
        mask
    ][:, mask] @ prod[mask]
    return smoothed


def read_npy(file):
    try:
        data = np.load(file, allow_pickle=True).item()
    except ValueError:
        data = np.load(file, allow_pickle=True)

    return data


# def pressure_thickness(p):
#    logp = np.log(p)
#    logp_edges = np.empty(len(p) + 1)
#
#    logp_edges[1:-1] = 0.5 * (logp[:-1] + logp[1:])
#    logp_edges[0] = logp[0] + 0.5 * (logp[0] - logp[1])
#    logp_edges[-1] = logp[-1] + 0.5 * (logp[-1] - logp[-2])
#
#    p_edges = np.exp(logp_edges)
#    dp = np.abs(np.diff(p_edges))
#
#    return dp


def pressure_thickness(p: np.ndarray) -> np.ndarray:
    """
    Compute the pressure thickness Δp for a pressure grid p.
    Handles boundaries robustly using logarithmic spacing.

    :param p: 1D array of pressures (Pa or hPa), decreasing or increasing
    :return: Δp array of same length as p
    """
    p = np.asarray(p * 1e2)

    # Ensure pressure decreases with index (common in satellite grids)
    if p[0] < p[-1]:
        p = p[::-1]  # flip if needed

    # Compute log-pressure edges for each layer
    logp = np.log(p)
    logp_edges = np.zeros(len(p) + 1)

    # Interior edges: midpoint between levels
    logp_edges[1:-1] = 0.5 * (logp[:-1] + logp[1:])

    # Boundaries: mirror first/last spacing (robust)
    # logp_edges[0] = logp[0] + 0.5 * (logp[0] - logp[1])
    # logp_edges[-1] = logp[-1] - 0.5 * (logp[-2] - logp[-1])

    logp_edges[0] = logp[0] + 0.5 * (logp[0] - logp[1])  # bottom edge (largest p)
    logp_edges[-1] = logp[-1] - 0.5 * (logp[-2] - logp[-1])  # top edge (smallest p)

    # Convert back to pressure
    p_edges = np.exp(logp_edges)

    # Layer thickness Δp = difference between edges
    dp = np.abs(np.diff(p_edges))

    return dp


def pressure_region_weights(p: NDArray, pmax: float, pmin: float) -> NDArray:
    """
    Rodgers OEM weight vector for pressure-averaged quantity,
    using ONLY pressure centers (monotonically decreasing).


    :param p: Pressure array
    :param pmax: Bottom of slice
    :param pmin: Top of slice
    :raises ValueError:
    :return: Weights
    """
    p = np.asarray(p)

    # ---- build edges from centers (geometric midpoints: log-grid safe)
    edges = np.zeros(len(p) + 1)
    edges[1:-1] = np.sqrt(p[:-1] * p[1:])
    edges[0] = p[0] ** 2 / edges[1]
    edges[-1] = p[-1] ** 2 / edges[-2]

    # ---- Rodgers overlap weighting
    N = len(p)
    w = np.zeros(N)

    p_low = min(pmin, pmax)
    p_high = max(pmin, pmax)

    for i in range(N):
        lo = max(min(edges[i], edges[i + 1]), p_low)
        hi = min(max(edges[i], edges[i + 1]), p_high)

        w[i] = max(0.0, np.log(hi) - np.log(lo))

    if w.sum() == 0:
        raise ValueError("No overlap with pressure region.")

    return w / w.sum()


def make_weighted_mean(data: Dict, pmax: float, pmin: float) -> NDArray:
    """Function to get the pressure weighted mean from a region in the atmosphere

    :param data: Dictionary with retrieval data
    :param s: Start of slice
    :param e: End of slice
    :return: Array with pressure wighted means from s to e on all data
    """
    means = []
    for v in data.values():
        try:
            p = v["pgrid"]
            vmr = v["x_phys"]
            # dp = pressure_thickness(p)
            # w = np.zeros(len(vmr))
            # w[s:e] = dp[s:e] / np.sum(dp[s:e])
            w = pressure_region_weights(p=p, pmax=pmax, pmin=pmin)
            vmr_mean = w @ vmr
            means.append(vmr_mean)
        except KeyError:
            p = v["p_interp"]
            vmr = v["O3_interp_smooth"]
            interped = v["O3_interp"]
            mask = np.isfinite(interped)
            # dp = pressure_thickness(p)
            # w = np.zeros(len(vmr))
            # w[s:e] = dp[s:e] / np.sum(dp[s:e])
            w = pressure_region_weights(p=p, pmax=pmax, pmin=pmin)
            vmr_mean = w[mask] @ vmr
            means.append(vmr_mean)

    return np.array(means)


def poly4_odr(beta, x):
    return beta[0] + beta[1] * x + beta[2] * x**2 + beta[3] * x**3 + beta[4] * x**4


def polynomial_uncertainty(x, pcov):
    # Build Jacobian matrix
    J = np.vstack([np.ones_like(x), x, x**2, x**3, x**4]).T

    # Propagate covariance
    var = np.sum(J @ pcov * J, axis=1)
    sigma = np.sqrt(var)

    return sigma


def fit_n2o_o3(
    x: NDArray,
    y: NDArray,
    errx: NDArray,
    erry: NDArray,
    filename: str,
    xmin=None,
    xmax=None,
) -> SimpleNamespace:
    xs = (x - x.mean()) / x.std()
    ys = (y - y.mean()) / y.std()
    errxs = errx / x.std()
    errys = erry / y.std()
    ddir = get_downloadsdir()
    savepath = ddir / filename

    data = RealData(x=xs, y=ys, sx=errxs, sy=errys)
    model = Model(poly4_odr)
    beta0 = [1, 0, 0, 0, 0]
    odr = ODR(data, model, beta0=beta0)
    output = odr.run()
    popt = output.beta
    pcov = output.cov_beta

    results = SimpleNamespace(
        params=popt,
        cov=pcov,
        mux=x.mean(),
        muy=y.mean(),
        sx=x.std(),
        sy=y.std(),
    )
    if xmin is None or xmax is None:
        xfit = np.linspace(x.min(), x.max(), 400)
    else:
        xfit = np.linspace(xmin, xmax, 400)
    xfit_s = (xfit - x.mean()) / x.std()
    yfit_s = poly4_odr(popt, xfit_s)
    yfit = (y.std() * yfit_s) + y.mean()
    sfit_s = polynomial_uncertainty(xfit_s, pcov)
    sfit = y.std() * sfit_s

    fit = SimpleNamespace(xfit=xfit, yfit=yfit, sfit=sfit)

    np.savez_compressed(
        file=savepath,
        params=results.params,
        cov=results.cov,
        mux=results.mux,
        muy=results.muy,
        sx=results.sx,
        sy=results.sy,
    )
    return results, fit


def binning(
    xdata, ydata, xerr, yerr, n_bins=200, core_percentile=(50, 50), min_bin_points=10
):
    quantiles = np.linspace(0, 1, n_bins + 1)
    bins = np.quantile(xdata, quantiles)
    indices = np.digitize(xdata, bins)

    keep = np.zeros_like(ydata, dtype=bool)

    x_centers = []
    y_median = []
    x_bin_err = []
    y_bin_err = []

    p_low, p_high = core_percentile

    for i in range(1, len(bins)):
        m = indices == i
        if np.sum(m) < min_bin_points:
            print("here")
            continue

        lo = np.percentile(ydata[m], p_low)
        hi = np.percentile(ydata[m], p_high)

        core = m & (ydata >= lo) & (ydata <= hi)
        keep |= core

        x_bin = xdata[core]
        y_bin = ydata[core]
        xerr_bin = xerr[core]
        yerr_bin = yerr[core]

        N = len(x_bin)
        if N == 0:
            continue

        # central values
        x_c = np.median(x_bin)
        y_c = np.median(y_bin)

        # measurement uncertainty
        x_meas = np.sqrt(np.sum(xerr_bin**2)) / N
        y_meas = np.sqrt(np.sum(yerr_bin**2)) / N

        # natural scatter
        x_scatter = np.std(x_bin) / np.sqrt(N)
        y_scatter = np.std(y_bin) / np.sqrt(N)

        # combined bin error
        x_err_bin = np.sqrt(x_meas**2 + x_scatter**2)
        y_err_bin = np.sqrt(y_meas**2 + y_scatter**2)

        x_centers.append(x_c)
        y_median.append(y_c)
        x_bin_err.append(x_err_bin)
        y_bin_err.append(y_err_bin)

    return (
        np.array(x_centers),
        np.array(y_median),
        np.array(x_bin_err),
        np.array(y_bin_err),
    )


def match_tracers(xdata, ydata):
    ydatapath = get_downloadsdir() / "O3_tracers_screened_matched.npy"
    xdatapath = get_downloadsdir() / "N2O_tracers_screened_matched.npy"
    pressurepath = get_datadir() / "m2pres.npz"

    vortexpath = get_downloadsdir() / "dmpdata.npy"
    vortex = np.load(vortexpath, allow_pickle=True).item()
    vortexdts = [dt for dt in vortex.keys()]

    # xdt = np.array([dt for dt in xdata])
    # ydt = np.array([dt for dt in ydata])
    # intersect = []
    # current = start

    # while current <= stop:
    #    intersect.append(current)
    #    current += timedelta(days=1)

    # if len(xdt) <= len(ydt):
    #    mlsdts = xdt

    # else:
    #    mlsdts = ydt

    # mlsdates = np.array([dt.date() for dt in mlsdts])

    # matchx = {}
    # matchy = {}
    # xpres = None
    # ypres = None

    # for d in intersect:
    #    mlsflag = d in mlsdates

    #    if mlsflag:
    #        mlsindex = np.where(mlsdates == d)[0]
    #        mlsk = mlsdts[mlsindex]
    #        for dt in mlsk:
    #            try:
    #                matchy[dt] = ydata[dt]
    #                matchx[dt] = xdata[dt]
    #                if xpres is None or ypres is None:
    #                    xpres = xdata[dt]["pressure"]
    #                    ypres = ydata[dt]["pressure"]
    #            except KeyError:
    #                continue

    matchx = {}
    matchy = {}
    pottemp = {}
    vortexmask = {}
    xpres = None
    ypres = None

    for dt in vortexdts:
        try:
            matchx[dt] = xdata[dt]
            matchy[dt] = ydata[dt]
            pottemp[dt] = vortex[dt]["theta_interp"]
            vortexmask[dt] = vortex[dt]["edgemask"]
            if xpres is None or ypres is None:
                xpres = xdata[dt]["pressure"]
                ypres = ydata[dt]["pressure"]
        except KeyError:
            continue

    xdt = np.array([dt for dt in matchx.keys()])
    ydt = np.array([dt for dt in matchy.keys()])
    if len(xdt) <= len(ydt):
        dts = xdt
    else:
        dts = ydt

    dts = ydt
    dct = {}
    # pres_thetas = np.load(pressurepath, allow_pickle=True)
    # ptarget = pres_thetas["pressure"]
    ptarget = np.load(pressurepath)["pressure"]
    logtgt = np.log(ptarget)

    for dt in dts:
        xsource = xpres
        ysource = ypres

        xvalcoarse = matchx[dt]["N2O"]
        xerrcoarse = matchx[dt]["precision"]
        yvalcoarse = matchy[dt]["O3"]
        yerrcoarse = matchy[dt]["precision"]

        logsrcx = np.log(xsource)
        logsrcy = np.log(ysource)

        interpx = interp1d(
            logsrcx,
            xvalcoarse,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        interperrx = interp1d(
            logsrcx,
            xerrcoarse,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

        interpy = interp1d(
            logsrcy,
            yvalcoarse,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        interperry = interp1d(
            logsrcy,
            yerrcoarse,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

        matchx[dt]["N2O_interp"] = interpx(logtgt)
        matchx[dt]["precision_interp"] = interperrx(logtgt)
        matchy[dt]["O3_interp"] = interpy(logtgt)
        matchy[dt]["precision_interp"] = interperry(logtgt)

        vmask = vortexmask[dt]
        print(ptarget[vmask])

        yval = matchy[dt]["O3_interp"][vmask]
        yerr = matchy[dt]["precision_interp"][vmask]
        xval = matchx[dt]["N2O_interp"][vmask]
        xerr = matchx[dt]["precision_interp"][vmask]

        mask = (yval > 0) & (xval > 0) & (yerr > 0) & (xerr > 0)

        dct[dt] = SimpleNamespace(
            xval=xval[mask],
            yval=yval[mask],
            xerr=xerr[mask],
            yerr=yerr[mask],
        )

    np.save(xdatapath, matchx, allow_pickle=True)
    np.save(ydatapath, matchy, allow_pickle=True)
    return dct


def propagate_uncertainty_mls(data: dict, pmax: int, pmin: int) -> NDArray:
    variances = []
    for v in data.values():
        p = v["p_interp"]
        sig2 = (v["precision_interp"] * 1e6) ** 2
        # dp = pressure_thickness(p)
        # w = np.zeros(len(p))
        # w[s:e] = dp[s:e] / np.sum(dp[s:e])
        w = pressure_region_weights(p=p, pmax=pmax, pmin=pmin)
        D = np.diag(sig2)
        variances.append(w.T @ D @ w)

    return np.array(variances)


def propagate_uncertainty_mira2(data: Dict, pmax: float, pmin: float) -> NDArray:
    variances = []
    for v in data.values():
        p = v["pgrid"]
        # dp = pressure_thickness(p)
        # w = np.zeros(len(p))
        # w[s:e] = dp[s:e] / np.sum(dp[s:e])
        w = pressure_region_weights(p=p, pmax=pmax, pmin=pmin)
        covar = v["S_phys"]
        sig2 = w.T @ covar @ w
        variances.append(sig2)

    return np.array(variances)
