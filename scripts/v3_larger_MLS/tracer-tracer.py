import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import plotly.graph_objects as go
import sys
from datetime import date
from scipy.optimize import curve_fit
from cmcrameri import cm
from scipy.odr import RealData, Model, ODR

from ozone.io import get_downloadsdir, get_egdefiles, get_exportdir
from ozone.utils import parse_edgefile, filter_edgedata
from ozone.analysis import get_period, poly4_odr, fit_n2o_o3
from ozone._const import COLORS

PLEV = 70


def fit_func(x, a0, a1, a2, a3, a4):
    return a0 + x * a1 + a2 * x**2 + a3 * x**3 + a4 * x**4


def poly4_odr(beta, x):
    return beta[0] + beta[1] * x + beta[2] * x**2 + beta[3] * x**3 + beta[4] * x**4


mlsO3fp = get_downloadsdir() / "v3_larger_MLS" / "MLS_O3_screened.npy"
mlsN2Ofp = get_downloadsdir() / "v3_larger_MLS" / "MLS_N2O_screened.npy"
edgefp = get_egdefiles("/home/ric/Data/edge")
mlsO3 = np.load(mlsO3fp, allow_pickle=True).item()
mlsN2O = np.load(mlsN2Ofp, allow_pickle=True).item()


dayO3 = mlsO3
dayN2O = mlsN2O
n2odt = [dt for dt in dayN2O.keys()]
dayO3 = {dt: dayO3[dt] for dt in n2odt}

mlsdates = np.array([k.date() for k in dayO3.keys()])
mlsdts = np.array([k for k in dayO3.keys()])
matchO3 = {}
matchN2O = {}


edgedata0 = filter_edgedata(parse_edgefile(edgefp[0]), 2)
edgedata1 = filter_edgedata(parse_edgefile(edgefp[1]), 2)
edgedata3 = filter_edgedata(parse_edgefile(edgefp[3]), 2)
intersect = list(set(edgedata0.date) & set(edgedata1.date) & set(edgedata3.date))

for d in intersect:
    mlsflag = d in mlsdates

    if mlsflag:
        mlsindex = np.where(mlsdates == d)[0]
        mlsk = mlsdts[mlsindex]
        for dt in mlsk:
            matchO3[dt] = dayO3[dt]
            matchN2O[dt] = dayN2O[dt]


for val in mlsO3.values():
    o3pres = val["pressure"]

for val in mlsN2O.values():
    n2opres = val["pressure"]


o3pres = np.array(o3pres)
n2opres = np.array(n2opres)

o3msk = o3pres <= PLEV
n2omsk = n2opres <= PLEV

o3lev = o3pres[o3msk][0]
n2olev = n2opres[n2omsk][0]

o3idx = np.where(o3pres == o3lev)[0][0]
n2oidx = np.where(n2opres == n2olev)[0][0]

startdate = date(2019, 11, 15)
stopdate = date(2020, 1, 5)

o3dt = np.array([dt for dt in matchO3.keys()])
n2odt = np.array([dt for dt in matchN2O.keys()])

assert np.all([o3dt, n2odt])

dts = o3dt
o3val = []
o3uncert = []
n2oval = []
n2ouncert = []

scdts = []

for dt in dts:
    d = dt.date()
    if startdate <= d <= stopdate:
        # Append O3 values
        o3val.extend(matchO3[dt]["O3"][i] for i in [o3idx - 2, o3idx, o3idx + 2])
        o3uncert.extend(
            matchO3[dt]["precision"][i] for i in [o3idx - 2, o3idx, o3idx + 2]
        )

        # Append N2O values
        n2oval.extend(matchN2O[dt]["N2O"][i] for i in [n2oidx - 1, n2oidx, n2oidx + 1])
        n2ouncert.extend(
            matchN2O[dt]["precision"][i] for i in [n2oidx - 1, n2oidx, n2oidx + 1]
        )

        # Append corresponding dates
        scdts.extend([d] * 3)

o3val = np.array(o3val)
o3uncert = np.array(o3uncert)
n2oval = np.array(n2oval)
n2ouncert = np.array(n2ouncert)
scdts = sorted(scdts)


res, fit = fit_n2o_o3(x=n2oval, y=o3val, errx=n2ouncert, erry=o3uncert)

xfit = fit.xfit
yfit = fit.yfit
sfit = fit.sfit


fig = plt.figure(figsize=(13, 10))
gs = GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.errorbar(
    x=n2oval * 1e9,
    y=o3val * 1e6,
    xerr=n2ouncert * 1e9,
    yerr=o3uncert * 1e6,
    fmt="o",
    color=COLORS.red,
    alpha=0.3,
    label=r"$O_3/N_2O$",
    zorder=1,
)
ax.plot(xfit * 1e9, yfit * 1e6, color=COLORS.mantle, linewidth=2, label=r"$Fit$")
ax.fill_between(
    xfit * 1e9,
    y1=(yfit + sfit) * 1e6,
    y2=(yfit - sfit) * 1e6,
    color=COLORS.mantle,
    alpha=0.2,
    label=r"$\pm \sigma$",
)
ax.minorticks_on()
ax.grid(which="both", alpha=0.1)
ax.set_ylabel(r"$O_3$ [ppmv]", fontsize=14)
ax.set_xlabel(r"$N_2O$ [ppbv]", fontsize=14)
ax.tick_params(labelsize=13)
ax.legend()
ax.set_title(f"{scdts[0]} - {scdts[-1]}")
fig.savefig("scatter_and_fit.png")
