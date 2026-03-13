from ozone.io import get_downloadsdir, get_datadir, get_home_data, geted
from ozone._const import COLORS
from ozone.analysis import poly4_odr, get_period, polynomial_uncertainty
from datetime import datetime, date
from scipy.interpolate import interp1d

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def calculate_passive_ozone(start, stop, n2odata, fitparams, isoplet):
    o3passive = []
    o3passive_err = []
    thetalev = []
    plevs = []
    dts = []
    for dt, val in n2odata.items():
        d = dt.date()
        if start <= d <= stop:
            n2oval = val["N2O_interp"] * 1e9
            theta = val["theta_interp"]
            pressure = val["pressure_interp"]
            edgemask = val["edgemask"]
            n2omasked = n2oval[edgemask]
            thetamasked = theta[edgemask]
            pressuremasked = pressure[edgemask]
            params = fitparams["params"]

            cov = fitparams["cov"]
            xmean = fitparams["mux"]
            ymean = fitparams["muy"]
            sx = fitparams["sx"]
            sy = fitparams["sy"]

            try:
                i = np.where(n2oval[edgemask] <= isoplet)[0][0]
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
                plevs.append(pressuremasked[i])
            except IndexError:
                continue

    return o3passive, o3passive_err, thetalev, plevs, dts


def calculate_loss(start, stop, n2odata, o3data, m2data, fitparams, isoplet):
    loss = []
    loss_err = []
    loss_dts = []
    loss_theta = []
    loss_mira2 = []
    loss_mira2_dts = []
    loss_mira2_err = []
    indecies = []

    (
        o3passive,
        o3passive_err,
        theta_lev,
        plevs,
        dts,
    ) = calculate_passive_ozone(start, stop, n2odata, fitparams, isoplet)

    m2dates = np.array([k.date() for k in m2data])
    m2datetimes = np.array([k for k in m2O3])
    for dt, o3p, o3pe, thl, pl in zip(dts, o3passive, o3passive_err, theta_lev, plevs):
        val = o3data[dt]
        theta = val["theta_interp"]
        pressure = val["p_interp"]

        try:
            i = np.where(theta == thl)[0][0]
            err = val["precision_interp"][i] * 1e6
            o3pe = o3pe * 1e6
            var = err**2 + o3pe**2
            loss.append((o3p - val["O3_interp"][i]) * 1e6)
            loss_err.append(np.sqrt(var))
            loss_dts.append(dt)
            loss_theta.append(theta[i])
            d = dt.date()
            indecies = np.where(m2dates == d)[0][:]
            for m2dt in m2datetimes[indecies]:
                val = m2O3[m2dt]
                cov = val["S_phys"]
                dcov = np.diag(cov)
                m2err = np.sqrt(dcov[i])
                loss_mira2.append(o3p * 1e6 - val["x_phys"][i])
                loss_mira2_err.append(m2err + o3pe)
                loss_mira2_dts.append(m2dt)
        except IndexError:
            continue

    return loss, loss_err, dts, loss_theta, loss_mira2, loss_mira2_dts, loss_mira2_err


dmppath = get_downloadsdir() / "dmpdata.npy"
pressure = get_datadir() / "m2pres.npz"
o3path = get_home_data() / "MLS_O3_screened_matching.npy"
n2opath = get_home_data() / "MLS_N2O_screened_matching.npy"
fitpath = get_home_data() / "fit_MLS.npz"
mira2path = get_home_data() / "MIRA2_v3_screened_matching.npy"

fitparams = np.load(fitpath)
mlsO3 = np.load(o3path, allow_pickle=True).item()
mlsN2O = np.load(n2opath, allow_pickle=True).item()
m2O3 = np.load(mira2path, allow_pickle=True).item()
dmpdata = np.load(dmppath, allow_pickle=True).item()
pressure = np.load(pressure)["pressure"]
logptarget = np.log(pressure)

mlsO3 = get_period(mlsO3, "day")
mlsN2O = get_period(mlsN2O, "day")
m2O3 = get_period(m2O3, "day")

o3dt = [dt for dt in mlsO3]

mlsO3_inside = {}
mlsN2O_inside = {}

for dt in dmpdata:
    if dt in o3dt:
        theta = dmpdata[dt]["theta_interp"]
        edgemask = dmpdata[dt]["edgemask"]
        mlsO3_inside[dt] = mlsO3[dt]
        mlsO3_inside[dt]["theta_interp"] = theta
        mlsO3_inside[dt]["edgemask"] = edgemask

        ptarget = mlsN2O[dt]["pressure"]
        logpsource = np.log(ptarget)
        interp = interp1d(
            logpsource,
            mlsN2O[dt]["N2O"],
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        n2o_interp = interp(logptarget)

        mlsN2O_inside[dt] = mlsN2O[dt]
        mlsN2O_inside[dt]["theta_interp"] = theta
        mlsN2O_inside[dt]["N2O_interp"] = n2o_interp
        mlsN2O_inside[dt]["pressure_interp"] = pressure
        mlsN2O_inside[dt]["edgemask"] = edgemask

np.save("MLS_N2O_matched_inside.npy", mlsN2O_inside, allow_pickle=True)
np.save("MLS_O3_matched_inside.npy", mlsO3_inside, allow_pickle=True)

start = date(2020, 1, 1)
stop = date(2020, 4, 15)
gs = GridSpec(3, 1)
fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

for i, iso in enumerate([30, 100]):
    loss, err, dts, losstheta, lossmira2, lossmira2_dts, lossmira2_err = calculate_loss(
        start=start,
        stop=stop,
        n2odata=mlsN2O_inside,
        o3data=mlsO3_inside,
        m2data=m2O3,
        fitparams=fitparams,
        isoplet=iso,
    )
    mlslosses = {}
    mlserrors = {}

    m2losses = {}
    m2errors = {}
    theta = np.round(np.mean([t for t in losstheta]))

    mlsloss = {}

    for l, e, dt in zip(loss, err, dts):
        d = dt.date()
        if d not in mlslosses:
            mlslosses[d] = [l]
            mlserrors[d] = [e]
        else:
            mlslosses[d].append(l)
            mlserrors[d].append(e)

    for l, e, dt in zip(lossmira2, lossmira2_err, lossmira2_dts):
        d = dt.date()
        if d not in m2losses:
            m2losses[d] = [l]
            m2errors[d] = [e]
        else:
            m2losses[d].append(l)
            m2errors[d].append(e)

    mlsmeanloss = np.array([np.mean(v) for v in mlslosses.values()])
    mlsmeanerr = np.array([np.mean(v) for v in mlserrors.values()])
    mlsdat = [key for key in mlslosses.keys()]
    mlserr = np.array(err).flatten()

    m2meanloss = np.array([np.mean(v) for v in m2losses.values()])
    m2dat = [key for key in m2losses.keys()]
    m2meanerr = np.array([np.mean(v) for v in m2errors.values()])
    mlsloss[iso] = {"dates": mlsdat, "loss": mlsmeanloss, "loss_err": mlsmeanerr}

    axes[i].set_ylim((-5, 5))
    axes[i].minorticks_on()
    axes[i].grid(which="both", alpha=0.1)

    axes[i].fill_between(
        x=mlsdat,
        y1=mlsmeanloss + mlsmeanerr,
        y2=mlsmeanloss - mlsmeanerr,
        color=COLORS.mantle,
        alpha=0.2,
    )
    axes[i].set_xlim((start, stop))
    axes[i].hlines(xmin=start, xmax=stop, y=0, color=COLORS.red, alpha=0.5)
    if i == 3:
        axes[i].set_xlabel("Date")
    axes[i].set_ylabel("Ozone loss [ppmv]")
    axes[i].set_title(f"Ozone loss from MLS data over Kiruna at ~{theta} K")
    axes[i].scatter(
        mlsdat,
        mlsmeanloss,
        color=COLORS.mantle,
        label=r"$\Delta O_3 \pm \sigma$ (MLS)",
    )
    axes[i].plot(
        mlsdat,
        mlsmeanloss,
        color=COLORS.mantle,
    )
    axes[i].fill_between(
        x=m2dat,
        y1=m2meanloss + m2meanerr,
        y2=m2meanloss - m2meanerr,
        color=COLORS.red,
        alpha=0.2,
    )
    axes[i].scatter(
        m2dat,
        m2meanloss,
        color=COLORS.red,
        label=r"$\Delta O_3 \pm \sigma$ (MIRA2)",
    )
    axes[i].plot(
        m2dat,
        m2meanloss,
        color=COLORS.red,
    )
    axes[i].legend()


fig.savefig("mls_loss_kir.pdf")
np.save(get_home_data() / "mlsloss.npy", mlsloss, allow_pickle=True)
