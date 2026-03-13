import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

from ozone.io import get_downloadsdir, get_datadir
from ozone.analysis import poly4_odr, get_period
from collections import defaultdict


from dataclasses import dataclass
from scipy.interpolate import interp1d
from datetime import date
from cmcrameri import cm

from ozone._const import COLORS


@dataclass
class CoincidentData:
    pressure: np.ndarray
    retrieved: np.ndarray
    err: np.ndarray
    date: date


@dataclass
class OzoneData:
    value: np.ndarray
    error: np.ndarray
    dt: np.ndarray


def interp(mlsdata, prod):
    ptarget = np.load(get_datadir() / "m2pres.npz", allow_pickle=True)["pressure"]
    logtarget = np.log(ptarget)
    for val in mlsdata.values():
        psource = val["pressure"]
        logsource = np.log(psource)
        datasource = val[prod]
        errsource = val["precision"]

        datainterp = interp1d(
            logsource,
            datasource,
            bounds_error=False,
            fill_value=np.nan,
        )
        errinterp = interp1d(
            logsource,
            errsource,
            bounds_error=False,
            fill_value=np.nan,
        )

        val[f"{prod}_interp"] = datainterp(logtarget)
        val["precision_interp"] = errinterp(logtarget)
        val["pressure_interp"] = ptarget
    return mlsdata


def read_data(source) -> dict(CoincidentData):
    ddir = get_downloadsdir()
    data = {}
    if source == "MIRA2":
        fp = ddir / "v3_larger_MLS" / "MIRA2_v3_screened_matching.npy"
        dct = np.load(fp, allow_pickle=True).item()
        dct = get_period(dct, "day")
        for dt, val in dct.items():
            d = dt.date
            var = np.diag(val["S_phys"])
            err = np.sqrt(var)
            pressure = val["pgrid"] / 1e2
            ret = val["x_phys"]
            data[dt] = CoincidentData(pressure=pressure, retrieved=ret, err=err, date=d)

    elif source == "O3" or source == "N2O":
        fp = ddir / "v3_larger_MLS" / f"MLS_{source}_screened_matching.npy"
        dct = np.load(fp, allow_pickle=True).item()
        dct = get_period(dct, "day")
        dct = interp(dct, source)
        grouped = defaultdict(list)
        atmovar = dict()
        for dt, val in dct.items():
            grouped[dt.date()].append(val[f"{source}_interp"])

        for dt, val in grouped.items():
            var = np.var(val, axis=0)
            atmovar[dt] = var

        for dt, val in dct.items():
            d = dt.date()
            avar = atmovar[d]
            totvar = val["precision_interp"] ** 2 + avar
            err = np.sqrt(totvar)
            data[dt] = CoincidentData(
                pressure=val["pressure_interp"],
                retrieved=val[f"{source}_interp"],
                err=err,
                date=d,
            )
    else:
        fp = ddir / "v3_larger_MLS" / f"MLS_{source}_screened.npy"
        dct = np.load(fp, allow_pickle=True).item()
        dct = get_period(dct, "day")
        grouped = defaultdict(list)
        atmovar = dict()

        for dt, val in dct.items():
            data[dt] = CoincidentData(
                pressure=val["pressure"],
                retrieved=val[f"{source}"],
                err=np.nan,
                date=np.nan,
            )

    return data


def make_cf(data):
    cf = []
    for val in data.values():
        ret = val.retrieved[6:14]
        ret[ret < 0] = 0
        cf.append(ret * 1e9)

    return np.array(cf)


def get_O3(data, plev):
    errs = []
    rets = []
    dts = []
    for dt, val in data.items():
        dts.append(dt)
        i = np.where(val.pressure >= plev)[0][-1]
        errs.append(val.err[i])
        rets.append(val.retrieved[i])

    return OzoneData(value=np.array(rets), error=np.array(errs), dt=np.array(dts))


dates = np.load(get_downloadsdir() / "dates.npz", allow_pickle=True)
pressure = np.load(get_datadir() / "m2pres.npz", allow_pickle=True)["pressure"]
inside = dates["dates"]

m2O3 = read_data("MIRA2")
m2O3lev = get_O3(m2O3, 40)

mlsO3 = read_data("O3")
mlsO3lev = get_O3(mlsO3, 40)

mlsN2O = read_data("N2O")
mlsClO = read_data("ClO")

m2O3cf = make_cf(data=m2O3)
mlsClOcf = make_cf(data=mlsClO)
mlsClOdts = [dt for dt in mlsClO.keys()]
clopres = np.mean([val.pressure for val in mlsClO.values()], axis=0)[6:14]
m2O3dts = [dt for dt in m2O3.keys()]

fig = plt.figure(figsize=(15, 8))
gs = GridSpec(3, 2, width_ratios=[20, 1], height_ratios=[1, 1, 1], wspace=0.02)
X, Y = np.meshgrid(mlsClOdts, clopres)
upper = fig.add_subplot(gs[0, 0])
middle = fig.add_subplot(gs[1, 0])
bottom = fig.add_subplot(gs[2, 0])
cax = fig.add_subplot(gs[0, 1])

upper.set_yscale("log")
upper.set_ylabel("Pressure [hPa]")
upper.invert_yaxis()
cf = upper.contourf(X, Y, mlsClOcf.T, cmap=cm.vik)
cb = fig.colorbar(cf, cax)
cb.set_label(r"$ClO$ / ppbv")

xlims = upper.get_xlim()

middle.minorticks_on()
middle.fill_between(
    x=m2O3lev.dt,
    y1=m2O3lev.value + m2O3lev.error,
    y2=m2O3lev.value - m2O3lev.error,
    color=COLORS.mantle,
    alpha=0.1,
)
middle.scatter(
    x=m2O3lev.dt, y=m2O3lev.value, color=COLORS.mantle, label=r"MIRA2 $\pm \sigma$", s=3
)
middle.errorbar(
    x=mlsO3lev.dt,
    y=mlsO3lev.value * 1e6,
    yerr=mlsO3lev.error * 1e6,
    color=COLORS.red,
    markersize=2,
    fmt="o",
    label=r"MLS $\pm \sigma$",
)
middle.scatter(
    x=inside,
    y=np.full_like(inside, 8),
    edgecolors=COLORS.mantle,
    color="white",
    s=10,
    label=r"Inside vortex ($Eql \geq 70$)",
)
middle.grid(which="both", alpha=0.1)
middle.set_xlim(xlims)
middle.set_ylabel(r"$O_3$ / ppmv (40 hPa)")
middle.legend(framealpha=1, fontsize=8)

m2O3lev = get_O3(m2O3, 60)
mlsO3lev = get_O3(mlsO3, 60)

bottom.minorticks_on()
bottom.fill_between(
    x=m2O3lev.dt,
    y1=m2O3lev.value + m2O3lev.error,
    y2=m2O3lev.value - m2O3lev.error,
    color=COLORS.mantle,
    alpha=0.1,
)
bottom.scatter(
    x=m2O3lev.dt, y=m2O3lev.value, color=COLORS.mantle, label=r"MIRA2 $\pm \sigma$", s=3
)
bottom.errorbar(
    x=mlsO3lev.dt,
    y=mlsO3lev.value * 1e6,
    yerr=mlsO3lev.error * 1e6,
    color=COLORS.red,
    markersize=2,
    fmt="o",
    label=r"MLS $\pm \sigma$",
)
bottom.scatter(
    x=inside,
    y=np.full_like(inside, 6),
    edgecolors=COLORS.mantle,
    color="white",
    s=10,
    label=r"Inside vortex ($Eql \geq 70$)",
)
bottom.grid(which="both", alpha=0.1)
bottom.set_xlim(xlims)
bottom.set_xlabel("Date")
bottom.set_ylabel(r"$O_3$ / ppmv (60 hPa)")
bottom.legend(framealpha=1, fontsize=8)

fig.savefig("ClO_with_comparison.png", transparent=True)
fig.savefig("ClO_with_comparison.pdf", transparent=True)
