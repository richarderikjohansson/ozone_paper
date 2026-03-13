import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmcrameri as cm
import scipy as sp
import sys
import pandas as pd
import cmcrameri.cm as cm

from datetime import datetime, date
from matplotlib.gridspec import GridSpec
from types import SimpleNamespace
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from ozone.io import get_downloadsdir, get_egdefiles
from ozone.utils import parse_edgefile, filter_edgedata
from ozone.analysis import (
    get_period,
    make_weighted_mean,
    propagate_uncertainty_mira2,
    propagate_uncertainty_mls,
)

VER = sys.argv[1]
PER = sys.argv[2]
PMAX = float(sys.argv[3])
PMIN = float(sys.argv[4])
COLORS = SimpleNamespace(
    {
        "overlay": "#6c7086",
        "mantle": "#181825",
        "red": "#d20f39",
        "peach": "#fe640b",
        "blue": "#1e66f5",
    }
)


def get_avk_offset(data):
    offset = []
    avk = data["avk"]
    altitude = data["zgrid"] / 1e3
    for i, row in enumerate(avk):
        mx = max(row)
        mxi = np.where(row == mx)[0][0]
        mxz = altitude[mxi]
        offset.append(altitude[i] - mxz)

    offset_filt = savgol_filter(offset, 10, 2)
    return offset_filt


# read files
def read_files():
    m2file = get_downloadsdir() / VER / f"MIRA2_{VER}_screened_matching.npy"
    mlsfile = get_downloadsdir() / VER / f"MLS_screened_matching{VER}.npy"
    m2full = np.load(m2file, allow_pickle=True).item()
    mlsfull = np.load(mlsfile, allow_pickle=True).item()
    m2 = get_period(m2full, PER)
    mls = get_period(mlsfull, PER)

    edgefiles = get_egdefiles("/home/ric/Data/edge/")
    edgedata = parse_edgefile(edgefiles[0])  # 550 K
    ed = filter_edgedata(edgedata, 1)  # strict filtering

    return m2, mls, ed


def abs_difference(m2, m2err, mls, mlserr):
    omega = m2
    mu = mls * 1e6
    m2var = m2err * m2err
    mlsvar = mlserr * mlserr
    err = np.sqrt(m2var + mlsvar)
    diff = omega - mu
    return diff, err


def plot_MIRA2_spec(m2single):
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(nrows=2, ncols=1, height_ratios=[2, 1])

    top = fig.add_subplot(gs[0, 0])
    bottom = fig.add_subplot(gs[1, 0])
    top.minorticks_on()

    for ax in fig.axes:
        ax.grid(which="both", alpha=0.1)

    top.plot(
        m2single["f"] / 1e9,
        m2single["y"],
        color=COLORS.overlay,
        label="Measurement",
    )
    top.plot(
        m2single["f"] / 1e9,
        m2single["yf"],
        color=COLORS.mantle,
        label="Retrieval Fit",
    )
    top.set_ylabel(r"$T_B$ [K]", fontsize=13)
    top.legend()

    bottom.plot(
        m2single["f"] / 1e9,
        m2single["residual"],
        color=COLORS.overlay,
        alpha=0.5,
        label="Residual",
    )
    xlims = bottom.get_xlim()
    bottom.hlines(
        y=0,
        xmin=xlims[0],
        xmax=xlims[-1],
        color=COLORS.overlay,
        linestyles="dashed",
        linewidth=1,
    )
    bottom.set_ylabel(r"$\Delta T_B$ [K]", fontsize=13)
    bottom.set_xlabel(r"$\nu$ [GHz]", fontsize=13)
    bottom.set_ylim([-2, 2])
    bottom.set_xlim(xlims)
    bottom.legend()

    fig.savefig(f"spec_{VER}_{PER}.pdf")
    plt.close()


def plot_MLS_MIRA2_comparison(m2single, mlssingle):
    m2pressure = m2single["pgrid"] / 100
    mlspressure = mlssingle["p_interp"] / 100
    m2x = m2single["x_phys"]
    mlsx = mlssingle["O3_interp_smooth"] * 1e6
    m2err = np.sqrt(m2single["ss"] + m2single["eo"])
    m2alt = m2single["zgrid"] / 1e3
    err_neg = m2single["x_phys"] - m2err
    err_pos = m2single["x_phys"] + m2err
    mlserr = mlssingle["precision_interp"]
    mlserrfix = []

    for i, err in enumerate(mlserr):
        if err < 0:
            mlserrfix.append(0)
        else:
            mlserrfix.append(err * 1e6)

    mlserrfix = np.array(mlserrfix)
    diff, err_diff = abs_difference(
        m2single["x_phys"],
        m2err,
        mlssingle["O3_interp_smooth"],
        mlserrfix,
    )
    err_diff_neg = diff - err_diff
    err_diff_pos = diff + err_diff

    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(nrows=1, ncols=2, wspace=0.1)
    left = fig.add_subplot(gs[0, 0])
    right = fig.add_subplot(gs[0, 1])
    left.set_yscale("log")
    left.minorticks_on()
    left.invert_yaxis()
    left.grid(which="both", alpha=0.1)
    left.fill_betweenx(
        m2single["pgrid"] / 100,
        err_neg,
        err_pos,
        color=COLORS.overlay,
        alpha=0.2,
        label=r"$\pm \sigma$",
    )
    left.plot(
        m2single["x_phys"],
        m2single["pgrid"] / 100,
        label="MIRA2",
        color=COLORS.mantle,
    )
    left.errorbar(
        mlssingle["O3_interp_smooth"] * 1e6,
        mlssingle["p_interp"] / 100,
        xerr=mlserrfix,
        linewidth=1,
        label="MLS (convolved)",
        color=COLORS.red,
        capsize=3,
        capthick=1,
    )
    left.plot(
        m2single["apriori"] * 1e6,
        m2single["pgrid"] / 100,
        linestyle="dashed",
        color=COLORS.overlay,
        label="apriori",
        linewidth=1,
    )

    left.set_xlim((-0.99, 10))
    left.legend(fontsize=8)
    left.set_xlabel(r"$O_3$ VMR [ppmv]", fontsize=14)
    left.set_ylabel("Pressure [hPa]", fontsize=14)
    left2 = left.twinx()

    # left2.set_ylabel("Altitude [km]", fontsize=14)
    left2.plot(m2single["x_phys"], m2single["zgrid"] / 1e3, linewidth=0)
    left2.tick_params(labelright=False, length=0)

    right.set_yscale("log")
    right.invert_yaxis()
    right.minorticks_on()
    err_pos = diff + err
    err_neg = diff - err

    right.fill_betweenx(
        m2single["pgrid"] / 100,
        err_diff_neg,
        err_diff_pos,
        label=r"$\pm \sigma$",
        color=COLORS.overlay,
        alpha=0.2,
    )
    ylims = right.get_ylim()
    right.vlines(
        x=0,
        ymin=ylims[0],
        ymax=ylims[-1],
        linestyles="dashed",
        color=COLORS.overlay,
        alpha=0.8,
    )
    right.plot(diff, m2single["pgrid"] / 100, color=COLORS.mantle, label="MIRA2 - MLS")
    right.set_xlim([-4.2, 4.2])
    right.legend(fontsize=8)
    right.set_xlabel(r"$\Delta O_3$ VMR [ppmv]", fontsize=14)
    # right.set_ylabel("Pressure [hPa]", fontsize=14)
    right.grid(which="both", alpha=0.1)
    right.tick_params(labelleft=False)

    right2 = right.twinx()
    right2.plot(diff, m2alt, linewidth=0)
    right2.set_ylabel("Altitude [km]", fontsize=14)
    for ax in fig.axes:
        ax.tick_params(which="both", labelsize=12)
    # plt.tight_layout()

    fig.savefig(f"profile_{VER}_{PER}.pdf")
    plt.close()


def vertical_resolution_from_AK(m2single, dz_fine=0.05):
    """
    Compute vertical resolution (FWHM) from averaging kernels.

    Parameters
    ----------
    A : ndarray (n_levels, n_levels)
        Averaging kernel matrix.
    z_levels : ndarray (n_levels,)
        Altitude corresponding to retrieval pressure grid.
        Units define output resolution (e.g. km).
    dz_fine : float
        Spacing of fine altitude grid (same units as z_levels).

    Returns
    -------
    resolution : ndarray (n_levels,)
        Vertical resolution at each retrieval level.
        NaN where FWHM cannot be determined.
    """

    A = m2single["avk"]
    z_levels = m2single["zgrid"] / 1e3
    n_levels = A.shape[0]

    # Fine regular altitude grid
    z_fine = np.arange(z_levels.min(), z_levels.max(), dz_fine)

    resolution = np.full(n_levels, np.nan)

    for i in range(n_levels):
        Ai = A[i, :].copy()

        # Skip empty rows
        if np.all(Ai == 0):
            continue

        # Normalize
        Ai /= Ai.max()

        # Interpolate AK row onto fine altitude grid
        f = interp1d(z_levels, Ai, kind="linear", bounds_error=False, fill_value=0.0)
        Ai_z = f(z_fine)

        # Find crossings of 0.5
        mask = Ai_z >= 0.5

        if not np.any(mask):
            continue

        idx = np.where(mask)[0]

        z1 = z_fine[idx[0]]
        z2 = z_fine[idx[-1]]

        resolution[i] = z2 - z1

    return resolution


def plot_AVK_vertres(m2single):
    zres = vertical_resolution_from_AK(m2single)
    offset = get_avk_offset(m2single)

    avk = m2single["avk"]
    pressure = m2single["pgrid"] / 100
    altitude = m2single["zgrid"] / 1000
    mr = m2single["mr"]

    n_levels = avk.shape[0]
    # my_map = mpl.colormaps.get_cmap("jet")
    my_map = cm.vik
    cmap_vector = np.linspace(0, 1, len(altitude))
    sm = plt.cm.ScalarMappable(
        cmap=my_map,
        norm=plt.Normalize(
            vmin=altitude[0],  # type: ignore
            vmax=altitude[-1],
        ),
    )

    fig = plt.figure(
        figsize=(12, 10),
    )
    gs = GridSpec(nrows=1, ncols=2, wspace=0.1)
    left = fig.add_subplot(gs[0, 0])
    left.invert_yaxis()
    left.minorticks_on()
    left.set_yscale("log")

    for i, row in enumerate(avk):
        color = my_map(cmap_vector[i])
        if i == 0:
            left.plot(3 * row, pressure, color=color, lw=1, label=r"3$\cdot$AVK")
        else:
            left.plot(3 * row, pressure, color=color, lw=1)

    # cb = plt.colorbar(sm, ax=left)
    # cb.set_label("Retrieval altitude [km]", fontsize=14)
    ylims = left.get_ylim()
    left.plot(mr, pressure, color=COLORS.mantle, label="MR")
    left.vlines(
        0.8,
        ylims[0],
        ylims[-1],
        linestyles="dashed",
        colors=COLORS.overlay,
        alpha=0.8,
    )
    left.legend(fontsize=8)
    left.grid(which="both", alpha=0.1)
    left.set_ylabel("Pressure [hPa]", fontsize=14)
    left.set_xlabel("AVK [-]", fontsize=14)

    right = fig.add_subplot(gs[0, 1])
    right.invert_yaxis()
    right.set_yscale("log")
    right.plot(zres, pressure, color=COLORS.mantle, label="Vertical resolution")
    right.plot(offset, pressure, color=COLORS.red, label="AVK offset")
    right.grid(which="both", alpha=0.1)
    right.set_ylim(ylims)
    right.vlines(
        x=0,
        ymin=ylims[0],
        ymax=ylims[-1],
        colors=COLORS.overlay,
        linestyles="dashed",
        alpha=0.8,
    )
    right.tick_params(labelleft=False)
    right.set_xlabel(r"$\Delta z$ [km]", fontsize=14)
    right.legend(fontsize=8)

    right2 = right.twinx()

    right2.plot(zres, altitude, linewidth=0)
    # right2.set_xlim(7, 23)
    right2.minorticks_on()
    right2.set_ylabel("Altitude [km]", fontsize=14)
    for ax in fig.axes:
        ax.tick_params(which="both", labelsize=12)

    fig.savefig(f"avk_vertres_{VER}_{PER}.pdf")
    plt.close()


def plot_relative_difference(m2, mls):
    pressure = np.mean([v["pgrid"] for v in m2.values()], axis=0)
    altitude = np.mean([v["zgrid"] for v in m2.values()], axis=0)
    msk = (pressure <= PMAX) * (pressure >= PMIN)
    ZMAX, ZMIN = round(altitude[msk][-1] / 1000), round(altitude[msk][0] / 1000)

    m2mean = make_weighted_mean(m2, PMAX, PMIN)
    m2sig = propagate_uncertainty_mira2(m2, PMAX, PMIN)
    m2date = [k.date() for k in m2.keys()]
    m2dt = [k for k in m2.keys()]
    m2df = pd.DataFrame(
        {
            "datetime": m2dt,
            "date": m2date,
            "m2mean": m2mean,
            "m2sig": m2sig,
        },
    )

    mlsmean = make_weighted_mean(mls, PMAX, PMIN)
    mlssig = propagate_uncertainty_mls(mls, PMAX / 100, PMIN / 100)
    mlsdate = [k.date() for k in mls.keys()]
    mlsdt = [k for k in mls.keys()]
    mlsdf = pd.DataFrame(
        {
            "datetime": mlsdt,
            "date": mlsdate,
            "mlsmean": mlsmean * 1e6,
            "mlssig": mlssig,
        }
    )

    m2dfmean_daily = (
        m2df.groupby(m2df["date"])["m2mean"].mean().reset_index(name="m2mean")
    )
    mlsdfmean_daily = (
        mlsdf.groupby(mlsdf["date"])["mlsmean"].mean().reset_index(name="mlsmean")
    )
    m2dfsig_daily = m2df.groupby(m2df["date"])["m2sig"].mean().reset_index(name="m2sig")
    mlsdfsig_daily = (
        mlsdf.groupby(mlsdf["date"])["mlssig"].mean().reset_index(name="mlssig")
    )

    merged_mean = pd.merge(m2dfmean_daily, mlsdfmean_daily, on="date", how="inner")
    merged_sig = pd.merge(m2dfsig_daily, mlsdfsig_daily, on="date", how="inner")

    dct = {
        "date": merged_mean["date"],
        "m2mean": merged_mean["m2mean"],
        "m2sig": merged_sig["m2sig"],
        "mlsmean": merged_mean["mlsmean"],
        "mlssig": merged_sig["mlssig"],
    }
    merged = pd.DataFrame(dct)
    rel, relsig, d = rel_diff(data=merged)
    dnumpy = d.to_numpy()

    fig = plt.figure(figsize=(18, 4))
    gs = GridSpec(1, 1)
    ax = fig.add_subplot(gs[0, 0])
    ax.errorbar(
        x=dnumpy,
        y=rel,
        yerr=relsig,
        capsize=3,
        color=COLORS.mantle,
        label=r"RD $\pm \sigma$",
    )

    # ax.minorticks_on()
    ax.grid(which="both", alpha=0.1)
    ax.set_title(
        f"Relative difference {round(PMAX / 100)} > p > {round(PMIN / 100)} hPa (~ {ZMIN} - {ZMAX} km)"
    )
    ax.fill_between(
        x=dnumpy, y1=10, y2=-10, color=COLORS.overlay, alpha=0.2, label=r"$\pm$ 10%"
    )
    xlims = ax.get_xlim()
    ax.hlines(
        y=0,
        xmin=xlims[0],
        xmax=xlims[-1],
        linestyles="dashed",
        colors=COLORS.overlay,
        alpha=0.8,
    )
    ax.set_xlabel("Date", fontsize=14)
    ax.set_ylabel("Relative difference [%]", fontsize=14)
    ax.tick_params(axis="both", labelsize=13)
    ax.set_ylim(-30, 30)
    ax.set_xlim(xlims)
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(f"relative_diff_{VER}_{PER}_{ZMIN}-{ZMAX}km.pdf")


def rel_diff(data):
    m2 = data["m2mean"]
    mls = data["mlsmean"]
    date = data["date"]
    m2sig = data["m2sig"]
    mlssig = data["mlssig"]

    A = 100 / m2
    B = (mls**2 / m2**2) * m2sig**2
    C = mlssig**2

    rel = 1e2 * (1 - (mls / m2))
    relsig = A * np.sqrt(B + C)

    return rel, relsig, date


def plot_differences(m2, mls):
    m2mean = make_weighted_mean(m2, PMAX, PMIN)


m2, mls, edge = read_files()

# date and times for MIRA2 and MLS
m2date = np.array([k.date() for k in m2.keys()])
m2dt = np.array([k for k in m2.keys()])
mlsdate = np.array([k.date() for k in mls.keys()])
mlsdt = np.array([k for k in mls.keys()])
m2maptime = {date: dt for date, dt in zip(m2date, m2dt)}
mlsmaptime = {date: dt for date, dt in zip(mlsdate, mlsdt)}


m2match = {}
mlsmatch = {}
for d in edge.date:
    try:
        m2dt = m2maptime[d]
        mlsdt = mlsmaptime[d]
        m2match[m2dt] = m2[m2dt]
        mlsmatch[mlsdt] = mls[mlsdt]

    except KeyError:
        continue

# extract MIRA2 and MLS data
posdate = date(2020, 4, 6)
m2key = m2maptime[posdate]
mlskey = mlsmaptime[posdate]

m2single = m2[m2key]
mlssingle = mls[mlskey]


plot = True
if plot:
    plot_MIRA2_spec(m2single)
    plot_MLS_MIRA2_comparison(m2single, mlssingle)
    plot_AVK_vertres(m2single)
    plot_relative_difference(m2match, mlsmatch)
