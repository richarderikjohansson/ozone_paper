from ozone.io import get_downloadsdir, get_datadir, get_home_data, get_egdefiles
from ozone.analysis import (
    poly4_odr,
    polynomial_uncertainty,
    filter_edgedata,
    parse_edgefile,
    get_period,
)
from datetime import date
import matplotlib.pyplot as plt

import numpy as np


def get_dates_from_dmpdata(mlsdata, dmpdata):
    dates = []
    for dt in dmpdata:
        if dt in mlsdata:
            dates.append(dt.date())

    uniquedates = set(dates)
    resulting_dates = sorted([d for d in uniquedates])
    return resulting_dates


def calculate_passive_ozone(inside_dates, mlsN2O, start, stop, fitparams, isoplet):
    o3passive = []
    o3passive_err = []
    dts = []
    index = []
    thetas = []
    params = fitparams["params"]

    cov = fitparams["cov"]
    xmean = fitparams["mux"]
    ymean = fitparams["muy"]
    sx = fitparams["sx"]
    sy = fitparams["sy"]

    for dt, val in mlsN2O.items():
        d = dt.date()
        if d in inside_dates and start <= d <= stop:
            theta = val["theta_interp"]
            n2oval = val["n2o_interp"] * 1e9
            mask = (theta >= 350) & (theta <= 650) & np.isclose(n2oval, isoplet, atol=3)
            if mask.sum() == 0:
                continue
            approx = n2oval[mask][0]
            thetas.append(theta[mask][0])
            i = np.where(n2oval == approx)
            dts.append(dt)
            approx /= 1e9
            approx_s = (approx - xmean) / sx
            o3ps = poly4_odr(params, approx_s)
            o3p = o3ps * sy + ymean
            o3passive.append(o3p)
            o3ps_err = polynomial_uncertainty(o3ps, cov)
            o3p_err = o3ps_err * sy
            o3passive_err.append(o3p_err)
            index.append(i)

    return (
        np.array(o3passive),
        np.array(o3passive_err),
        np.array(dts),
        np.array(index),
        np.array(thetas),
    )


def calculate_loss(o3passive, o3passive_err, dts, index, mdata, which="MLS"):
    dates = np.array([dt.date() for dt in dts])
    dats = []
    loss = []
    loss_err = []
    if which == "MIRA2":
        for dt, val in mdata.items():
            d = dt.date()
            if d in dates:
                theta = val["theta"]
                dateidx = np.where(dates == d)
                i = index[dateidx].flatten()
                o3p = np.mean(o3passive[dateidx])
                o3pe = np.mean(o3passive_err[dateidx])
                cov = val["S_phys"]
                dcov = np.diag(cov)
                o3me = np.mean(np.sqrt(dcov[i])) / 1e6
                o3m = np.mean(val["x_phys"][i]) / 1e6

                loss.append(o3m - o3p)
                loss_err.append(o3pe + o3me)
                dats.append(d)
    else:
        for dt, val in mdata.items():
            d = dt.date()
            if d in dates:
                dateidx = np.where(dates == d)
                i = index[dateidx].flatten()
                o3p = np.mean(o3passive[dateidx])
                o3pe = np.mean(o3passive_err[dateidx])

                o3me = np.mean(val["precision_interp"][i])
                o3m = np.mean(val["O3_interp"][i])
                loss.append(o3m - o3p)
                loss_err.append(o3pe + o3me)
                dats.append(d)

    return np.array(loss), np.array(loss_err), np.array(dats)


def get_dates_in_vortex(edgefiles, severity):
    f475 = edgefiles[1]
    f550 = edgefiles[0]
    f675 = edgefiles[3]
    edge475 = filter_edgedata(parse_edgefile(f475), severity=severity).date
    edge550 = filter_edgedata(parse_edgefile(f550), severity=severity).date
    edge675 = filter_edgedata(parse_edgefile(f675), severity=severity).date

    # common_dates = sorted(list(set(edge475) & set(edge550) & set(edge675)))
    # return np.array(common_dates)
    return np.array(edge550)


m2p = get_home_data() / "MIRA2_v3_screened_matching_w_theta.npy"
mlsN2Op = get_home_data() / "MLS_N2O_screened_matching_interp.npy"
mlsO3p = get_home_data() / "MLS_O3_screened_matching.npy"
dmpdatap = get_home_data() / "dmpdata_kir.npy"
mlsfitpath = get_home_data() / "fit_MLS.npz"
m2fitpath = get_home_data() / "fit_MIRA2.npz"
edgefiles = get_egdefiles("/home/ric/Data/edge/")

inside_dates = get_dates_in_vortex(edgefiles, 2)

m2O3 = np.load(m2p, allow_pickle=True).item()
mlsN2O = np.load(mlsN2Op, allow_pickle=True).item()
mlsO3 = np.load(mlsO3p, allow_pickle=True).item()
dmpdata = np.load(dmpdatap, allow_pickle=True).item()

m2O3 = get_period(m2O3, "day")
mlsN2O = get_period(mlsN2O, "day")
mlsO3 = get_period(mlsO3, "day")
fitmls = np.load(mlsfitpath)
fitm2 = np.load(m2fitpath)

dates = get_dates_from_dmpdata(mlsdata=mlsO3, dmpdata=dmpdata)
# inside_dates = dates

print(inside_dates)
start = date(2019, 10, 15)
stop = date(2020, 4, 30)
isoloss = {}
ds = {
    "MIRA2": {"fit": fitm2, "data": m2O3, "which": "MIRA2", "fname": "m2loss.npy"},
    "MLS": {"fit": fitmls, "data": mlsO3, "which": "MLS", "fname": "mlsloss.npy"},
}
for val in ds.values():
    fit = val["fit"]
    data = val["data"]
    which = val["which"]
    fname = val["fname"]

    for iso in [45, 97]:
        o3p, o3pe, dts, index, thetas = calculate_passive_ozone(
            inside_dates,
            mlsN2O,
            start,
            stop,
            fit,
            iso,
        )
        loss, losserr, dats = calculate_loss(o3p, o3pe, dts, index, data, which=which)
        m2loss = {}
        m2err = {}
        for l, e, d in zip(loss, losserr, dats):
            if d not in m2loss:
                m2loss[d] = [l]
                m2err[d] = [e]
            else:
                m2loss[d].append(l)
                m2err[d].append(e)
        meanloss = np.array([np.mean(v) for v in m2loss.values()])
        meanerr = np.array([np.mean(v) for v in m2err.values()])
        dat = [key for key in m2loss.keys()]
        isoloss[iso] = {
            "loss": meanloss,
            "dates": dat,
            "loss_err": meanerr,
            "theta": (thetas.min(), thetas.max()),
        }

    np.save(get_home_data() / fname, isoloss, allow_pickle=True)
