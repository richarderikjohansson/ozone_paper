from ozone.io import get_downloadsdir, get_home_data
from ozone.analysis import poly4_odr, get_period, polynomial_uncertainty
from datetime import date

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def calc_pottemp(t, p):
    kap = 0.286

    pot = t * (1e5 / p) ** kap
    return pot


def calculate_passive_ozone(start, stop, n2odata, fitparams, isolev):
    o3passive = []
    o3passive_err = []
    thetalev = []
    dts = []
    for dt, val in n2odata.items():
        d = dt.date()
        if start <= d <= stop:
            n2oval = val["N2O_interp"] * 1e9
            theta = val["theta"]
            edgemask = val["edgemask"]
            n2omasked = n2oval[edgemask]
            thetamasked = theta[edgemask]
            params = fitparams["params"]
            cov = fitparams["cov"]
            xmean = fitparams["mux"]
            ymean = fitparams["muy"]
            sx = fitparams["sx"]
            sy = fitparams["sy"]

            try:
                i = np.where(thetamasked >= isolev)[0][0]
                dts.append(dt)
                n2o50 = n2omasked[i] / 1e9
                n2o_50_s = (n2o50 - xmean) / sx
                o3ps = poly4_odr(params, n2o_50_s)
                o3p = o3ps * sy + ymean
                o3passive.append(o3p)
                o3ps_err = polynomial_uncertainty(o3ps, cov)
                o3p_err = o3ps_err * sy
                o3passive_err.append(o3p_err)
                thetalev.append(thetamasked[i])
            except IndexError:
                continue

    return o3passive, o3passive_err, thetalev, dts


dmpdatap = get_downloadsdir() / "dmpdata.npy"
mira2p = get_downloadsdir() / "v3_larger_MLS" / "MIRA2_v3_screened_matching.npy"
mlsn2op = "MLS_N2O_matched_inside.npy"
fitpath = get_downloadsdir() / "fit_filtered.npz"

fitparams = np.load(fitpath)
start = date(2020, 1, 15)
stop = date(2020, 3, 25)


mira2 = np.load(mira2p, allow_pickle=True).item()
dmpdata = np.load(dmpdatap, allow_pickle=True).item()
n2odata = np.load(mlsn2op, allow_pickle=True).item()
o3p, o3pe, tl, dts = calculate_passive_ozone(start, stop, n2odata, fitparams, 460)
dates = [k.date() for k in dts]

pressure = np.mean([v["pgrid"] for v in mira2.values()], axis=0)
logp = np.log(pressure)


mira2wtheta = {}
for k, v in mira2.items():
    logsrc = np.log(v["pmeas"])
    xsrc = v["tmeas"]
    mira2wtheta[k] = v

    interp = interp1d(
        logsrc,
        xsrc,
        bounds_error=False,
        fill_value=np.nan,
    )

    ttar = interp(logp)
    pot = calc_pottemp(ttar, pressure)
    mira2wtheta[k]["theta"] = pot

np.save("MIRA2_matched_w_theta.npy", mira2wtheta, allow_pickle=True)
