import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from datetime import date, timedelta
from pathlib import Path


from ozone.io import get_egdefiles, get_downloadsdir
from ozone.analysis import (
    make_weighted_mean,
    propagate_uncertainty_mira2,
    propagate_uncertainty_mls,
    get_period,
)
from ozone.utils import filter_edgedata, parse_edgefile

PERIOD = sys.argv[1]
PMAX = float(sys.argv[2])
PMIN = float(sys.argv[3])


def make_pressure():
    p_surface = 101325  # Pa (surface pressure)
    p_top = 1  # Pa (top of atmosphere)
    n_levels = 100  # number of grid points
    pressure = np.logspace(np.log10(p_surface), np.log10(p_top), n_levels)
    return pressure


def make_altitude(pgrid):
    R = 8.3144598
    T_b = 231
    M = 0.0289644
    g = 9.819
    pref = 96000
    href = 430
    return href - (R * T_b / (M * g)) * np.log(pgrid / pref)


def get_assetsdir(subdir):
    home = Path.home()
    return home / "Presentations" / "reveal.js" / "presentation" / subdir / "assets"


def rel_diff(data):
    m2 = data["m2mean"]
    mls = data["mlsmean"]
    m2sig = data["m2sig"]
    mlssig = data["mlssig"]

    A = 100 / m2
    B = (mls**2 / m2**2) * m2sig**2
    C = mlssig**2

    rel = 1e2 * (1 - (mls / m2))
    relsig = A * np.sqrt(B + C)

    return rel, relsig


def make_date_range():
    start = date(2019, 10, 1)
    end = date(2020, 5, 1)
    cur = start
    daterange = [cur]

    while cur < end:
        cur += timedelta(days=1)
        daterange.append(cur)

    return np.array(daterange)


def parse_data(fp, period):
    data = np.load(fp, allow_pickle=True).item()
    data_at_period = get_period(data, period)
    return data_at_period


def get_pressure_weighted_mean(data, which, pmax, pmin):
    match which:
        case "mira2":
            sigma = propagate_uncertainty_mira2(data, pmax, pmin)
            mean = make_weighted_mean(data, pmax, pmin)
        case "mls":
            sigma = propagate_uncertainty_mls(data, pmax, pmin)
            mean = make_weighted_mean(data, pmax, pmin)
            mean *= 1e6

    return mean, sigma


m2fp = get_downloadsdir() / "v3_larger_MLS" / "MIRA2_v3_screened_matching.npy"
mlsfp = get_downloadsdir() / "v3_larger_MLS" / "MLS_O3_screened_matching.npy"
pres_dir = get_assetsdir("coincident_measurements")

m2 = parse_data(m2fp, PERIOD)
mls = parse_data(mlsfp, PERIOD)
m2dates = np.array([k.date() for k in m2.keys()])
m2dts = np.array([k for k in m2.keys()])
mlsdates = np.array([k.date() for k in mls.keys()])
mlsdts = np.array([k for k in mls.keys()])
daterange = make_date_range()
efiles = get_egdefiles("/home/ric/Data/edge")

e0 = filter_edgedata(parse_edgefile(efiles[0]), 2)
e1 = filter_edgedata(parse_edgefile(efiles[1]), 2)
e3 = filter_edgedata(parse_edgefile(efiles[3]), 2)
e4 = filter_edgedata(parse_edgefile(efiles[4]), 2)
# edgedate = np.unique(np.concatenate([e0.date, e1.date]))

eqldata = np.load(get_downloadsdir() / "dates.npz", allow_pickle=True)
edgedate = eqldata["insidevort"]


m2match = {}
mlsmatch = {}

for d in edgedate:
    m2flag = d in m2dates
    mlsflag = d in mlsdates

    if m2flag and mlsflag:
        m2index = np.where(m2dates == d)[0]
        mlsindex = np.where(mlsdates == d)[0]
        m2k = m2dts[m2index[0]]
        mlsk = mlsdts[mlsindex[0]]
        m2match[m2k] = m2[m2k]
        mlsmatch[mlsk] = mls[mlsk]

m2dt = np.array([k for k in m2match.keys()])
m2mean, m2var = get_pressure_weighted_mean(m2match, "mira2", PMAX, PMIN)
m2sig = np.sqrt(m2var)
mlsmean, mlsvar = get_pressure_weighted_mean(mlsmatch, "mls", PMAX, PMIN)
mlssig = np.sqrt(mlsvar)
mlsdt = np.array([k for k in mlsmatch.keys()])
# pressure = np.mean([v["pgrid"] for v in m2match.values()], axis=0) / 1e2
# altitude = np.mean([v["zgrid"] for v in m2match.values()], axis=0) / 1e3

pressure = make_pressure()
altitude = make_altitude(pressure)

data = {"m2mean": m2mean, "mlsmean": mlsmean, "m2sig": m2sig, "mlssig": mlssig}
rel, relsig = rel_diff(data)

err_pos = m2mean + m2sig
err_neg = m2mean - m2sig
msk = (PMAX >= pressure) & (PMIN <= pressure)
altmin = np.round(altitude[msk][0], 0)
altmax = np.round(altitude[msk][-1], 0)


fig = make_subplots(rows=2, cols=1)
fig.update_layout(
    template="plotly_white",
    height=600,
    width=1600,
    title=f"Coincident MIRA2 and MLS measurements {PMAX / 100} > p > {PMIN / 100} hPa ({altmin / 1000}-{altmax / 1000} km)",
)
fig.add_trace(
    go.Scatter(
        x=m2dt,
        y=err_pos,
        showlegend=False,
        mode="lines",
        line=dict(width=0),
        legendgroup="uncert",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=m2dt,
        y=err_neg,
        showlegend=True,
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(0,0,0,0.2)",
        name="±σ",
        legendgroup="uncert",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=m2dt, y=m2mean, mode="markers", marker=dict(color="black"), showlegend=False
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=mlsdt, y=mlsmean, mode="markers", marker=dict(color="red"), name="MLS"
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=m2dt,
        y=np.full_like(m2dt, 10),
        showlegend=False,
        mode="lines",
        line=dict(width=0),
        legendgroup="lims",
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=m2dt,
        y=np.full_like(m2dt, -10),
        showlegend=True,
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(255,0,0,0.2)",
        name=r"10% limit",
        legendgroup="lims",
    ),
    row=2,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=m2dt,
        y=np.full_like(m2dt, 0),
        mode="lines",
        line=dict(dash="dash", color="gray"),
        showlegend=False,
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=m2dt,
        y=rel + relsig,
        showlegend=False,
        mode="lines",
        line=dict(width=0),
        legendgroup="uncert",
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=m2dt,
        y=rel - relsig,
        showlegend=False,
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(0,0,0,0.2)",
        name="±σ",
        legendgroup="uncert",
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=m2dt, y=rel, mode="markers", marker=dict(color="black"), showlegend=False
    ),
    row=2,
    col=1,
)
fig.update_yaxes(row=2, col=1, range=[-55, 55])
fig.write_html(pres_dir / f"coincident_inside_edge_{PMAX}-{PMIN}.html")

m2scat = [v["x_phys"] for v in m2match.values()]
mlsscat = [v["O3_interp_smooth"] * 1e6 for v in mlsmatch.values()]
