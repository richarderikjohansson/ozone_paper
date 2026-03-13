import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

from ozone.io import get_downloadsdir, get_egdefiles
from ozone.utils import parse_edgefile, filter_edgedata
from ozone.analysis import (
    get_period,
    make_weighted_mean,
    propagate_uncertainty_mira2,
    propagate_uncertainty_mls,
)


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


PMAX = float(sys.argv[1])
PMIN = float(sys.argv[2])
version = sys.argv[3]
period = sys.argv[4]

if version == "v3":
    m2v3f = get_downloadsdir() / "v3" / "MIRA2_v3_screened_matching.npy"
    mlsf = get_downloadsdir() / "v3" / "MLS_screened_matchingv3.npy"
elif version == "v4":
    m2v3f = get_downloadsdir() / "v4" / "MIRA2_v4_screened_matching.npy"
    mlsf = get_downloadsdir() / "v4" / "MLS_screened_matchingv4.npy"

# Edge data
files = get_egdefiles("/home/ric/Data/edge/")
edgedata = parse_edgefile(files[1])
ed = filter_edgedata(edgedata, 2)

# MIRA2
m2day = get_period(np.load(m2v3f, allow_pickle=True).item(), period)
m2dt, m2date = [], []
for dt in m2day.keys():
    m2dt.append(dt)
    m2date.append(dt.date())
# m2date = np.unique(m2date)
m2mr = [v["mr"] for v in m2day.values()]
m2pressure = np.mean([v["pgrid"] for v in m2day.values()], axis=0)

# MLS
mlsday = get_period(np.load(mlsf, allow_pickle=True).item(), period)
mlsdt, mlsdate = [], []
for dt in mlsday.keys():
    mlsdt.append(dt)
    mlsdate.append(dt.date())

m2date = np.array([k.date() for k in m2day.keys()])
m2dt = np.array([k for k in m2day.keys()])
mlsdate = np.array([k.date() for k in mlsday.keys()])
mlsdt = np.array([k for k in mlsday.keys()])
m2maptime = {date: dt for date, dt in zip(m2date, m2dt)}
mlsmaptime = {date: dt for date, dt in zip(mlsdate, mlsdt)}

m2mean = make_weighted_mean(m2day, PMAX, PMIN)
m2sig = propagate_uncertainty_mira2(m2day, PMAX, PMIN)
m2df = pd.DataFrame(
    {
        "datetime": m2dt,
        "date": m2date,
        "m2mean": m2mean,
        "m2sig": m2sig,
    },
)

mlsmean = make_weighted_mean(mlsday, PMAX, PMIN)
mlssig = propagate_uncertainty_mls(mlsday, PMAX / 100, PMIN / 100)
mlsdf = pd.DataFrame(
    {
        "datetime": mlsdt,
        "date": mlsdate,
        "mlsmean": mlsmean * 1e6,
        "mlssig": mlssig,
    }
)

m2dfmean_daily = m2df.groupby(m2df["date"])["m2mean"].mean().reset_index(name="m2mean")
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

fig = go.Figure()
fig.update_layout(
    template="plotly_white",
    height=800,
    width=600,
    title=f"Measurement response {version} ({period})",
)
fig.update_yaxes(autorange="reversed", type="log", title="Pressure [hPa]")
fig.update_xaxes(title="Measurement response [-]")
fig.update_yaxes(title="Pressure [hPa]")
for mr in m2mr:
    fig.add_trace(go.Scatter(x=mr, y=m2pressure / 100, showlegend=False))

fig.write_html(f"mr{version}_{period}.html")

rel_pos = rel + relsig
rel_neg = rel - relsig

mx = max(rel_pos)
mn = min(rel_neg)
ymax = mx + 5
ymin = mn - 5

fig = go.Figure()
fig.update_layout(
    template="plotly_white",
    height=400,
    width=1600,
    title=f"Relative difference {version} ({PMIN / 100} < p < {PMAX / 100}) ({period}) ~ 18 - 46 km",
)

fig.add_trace(
    go.Scatter(
        x=ed.date,
        y=np.full_like(ed.date, ymax - 2),
        mode="markers",
        marker=dict(symbol="circle-open", color="black"),
        name="Inside vortex (475 K)",
        showlegend=True,
    ),
)
fig.add_trace(
    go.Scatter(
        x=d,
        y=rel,
        mode="markers+lines",
        marker=dict(color="black", size=3),
        name="Relative difference",
        legendgroup="MIRA2",
        showlegend=True,
        error_y=dict(type="data", array=relsig, visible=True),
    ),
)
rel_lims = [ed.date[0], dnumpy[-1]]
fig.add_trace(
    go.Scatter(
        x=rel_lims,
        y=np.full_like(rel_lims, 10),
        showlegend=False,
        mode="lines",
        line=dict(width=0),
        legendgroup="uncert",
    ),
)
fig.add_trace(
    go.Scatter(
        x=rel_lims,
        y=np.full_like(rel_lims, -10),
        showlegend=True,
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(0,0,0,0.2)",
        name=r"10% limits",
        legendgroup="uncert",
    ),
)
fig.update_yaxes(range=[-40, 40], title="Relative difference (%)")
fig.update_xaxes(title="Date")
fig.write_html(f"relative_difference{version}_{PMAX / 100}-{PMIN / 100}_{period}.html")
