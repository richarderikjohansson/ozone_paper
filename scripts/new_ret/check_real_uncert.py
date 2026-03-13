import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ozone.io import get_downloadsdir, get_egdefiles
from ozone.analysis import (
    propagate_uncertainty_mira2,
    get_period,
    make_weighted_mean,
    pressure_thickness,
    propagate_uncertainty_mls,
    pressure_region_weights,
)
from ozone.utils import parse_edgefile, filter_edgedata
import numpy as np


def make_weighted_mean_mls(data, s, e):
    means = []
    for v in data.values():
        p = v["p_interp"]
        dp = pressure_thickness(p)
        vmr = v["O3_interp"]
        w = np.zeros(len(vmr))
        w[s:e] = dp[s:e] / np.sum(dp[s:e])
        vmr_mean = w @ vmr
        means.append(vmr_mean)

    return np.array(means)


def prop_mls(data, s, e):
    variances = []
    for v in data.values():
        p = v["p_interp"]
        dp = pressure_thickness(p)
        precision = v["precision_interp"] * 1e6
        w = np.zeros(len(precision))
        w[s:e] = dp[s:e] / np.sum(dp[s:e])
        precision_mean = w @ precision**2
        variances.append(precision_mean)

    return np.array(variances)


PMAX = 45e2
PMIN = 10e2

# EDGE file
files = get_egdefiles("/home/ric/Data/edge/")
edgedata = parse_edgefile(files[1])
ed = filter_edgedata(edgedata, 2)
#
# MLS
mlsfile = get_downloadsdir() / "v3" / "MLS_screened_matchingv3.npy"
mls = np.load(mlsfile, allow_pickle=True).item()
mlsday = get_period(mls, "day")
mlslats = [v["lat"] for v in mlsday.values()]
mlslons = [v["lon"] for v in mlsday.values()]
mlsdt = [dt for dt in mlsday.keys()]
mlsmean_smooth = make_weighted_mean(data=mlsday, pmax=PMAX, pmin=PMIN)
mlsvar = propagate_uncertainty_mls(data=mlsday, pmax=PMAX, pmin=PMIN)
precision = [v["precision_interp"] for v in mlsday.values()]
mlspres = np.mean([v["p_interp"] for v in mlsday.values()], axis=0)
msk = (mlspres < PMAX) & (mlspres > PMIN)
mlsstd = []
for pres in precision:
    mlsstd.append(np.mean(pres[msk] * 1e6))
mlsstd = np.array(mlsstd)
mls_pos = mlsmean_smooth * 1e6 + mlsstd
mls_neg = mlsmean_smooth * 1e6 - mlsstd


# MIRA2
mira2file = get_downloadsdir() / "v3" / "MIRA2_v3_screened_matching.npy"
data = np.load(mira2file, allow_pickle=True).item()
mira2day = get_period(data=data, period="day")
pressure = np.mean([v["pgrid"] for v in mira2day.values()], axis=0)
altitude = np.mean([v["zgrid"] for v in mira2day.values()], axis=0)
m2dt = [dt for dt in mira2day.keys()]
p = mira2day[m2dt[0]]["pgrid"]
mira2var = propagate_uncertainty_mira2(data=mira2day, pmax=PMAX, pmin=PMIN)
mira2mean = make_weighted_mean(data=mira2day, pmax=PMAX, pmin=PMIN)
mira2std = np.sqrt(mira2var)
for p, z in zip(pressure, altitude):
    print(p, z)

m2std_pos = mira2mean + mira2std
m2std_neg = mira2mean - mira2std

m2v4file = get_downloadsdir() / "MIRA2_v_4_screened.npy"
m2v4 = np.load(m2v4file, allow_pickle=True).item()
m2v4day = get_period(m2v4, period="day")
m2v4dt = [dt for dt in m2v4day.keys()]
m2v4mean = make_weighted_mean(m2v4day, pmax=PMAX, pmin=PMIN)
m2v4var = propagate_uncertainty_mira2(m2v4day, pmax=PMAX, pmin=PMIN)
m2v4std = np.sqrt(m2v4var)

m2v4_pos = m2v4mean + m2v4std
m2v4_neg = m2v4mean - m2v4std


pressure = mira2day[m2dt[0]]["pgrid"]
altitude = mira2day[m2dt[0]]["zgrid"]
for p, z in zip(pressure, altitude):
    print(p, z)


fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    column_titles=[
        f"{PMIN / 100} < p < {PMAX / 100} hPa - MIPAS apriori (fixed)",
        f"{PMIN / 100} < p < {PMAX / 100} hPa - ECMWF apriori (change on month)",
    ],
)
fig.update_layout(template="plotly_white", height=400, width=1600)
fig.add_trace(
    go.Scatter(
        x=m2dt,
        y=m2std_pos,
        showlegend=False,
        line=dict(width=0),
        legendgroup="uncert",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=m2dt,
        y=m2std_neg,
        showlegend=True,
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(0,0,0,0.2)",
        name=r"MIRA2 uncertainty",
        legendgroup="uncert",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=m2dt,
        y=mira2mean,
        mode="markers",
        marker=dict(color="black", size=3),
        name="MIRA2",
        legendgroup="MIRA2",
        showlegend=True,
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=mlsdt,
        y=mlsmean_smooth * 1e6,
        mode="lines",
        line=dict(color="red"),
        name="MLS (convolved)",
        legendgroup="MLS",
        showlegend=True,
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=ed.date,
        y=np.full_like(m2v4_pos, max(m2v4_pos) + 0.5),
        mode="markers",
        marker=dict(color="black", symbol="circle-open"),
        name="PV EDGE (475K)",
        legendgroup="edge",
        showlegend=True,
    ),
    row=1,
    col=1,
)

# row 2
mlsfile = get_downloadsdir() / "MLS_screenedv4_matching.npy"
mls = np.load(mlsfile, allow_pickle=True).item()
mlsday = get_period(mls, "day")
mlsdt = [dt for dt in mlsday.keys()]
mlsmean_smooth = make_weighted_mean(data=mlsday, pmax=PMAX, pmin=PMIN)
mlsvar = propagate_uncertainty_mls(data=mlsday, pmax=PMAX, pmin=PMIN)
precision = [v["precision_interp"] for v in mlsday.values()]
mlspres = np.mean([v["p_interp"] for v in mlsday.values()], axis=0)
msk = (mlspres < PMAX) & (mlspres > PMIN)
mlsstd = []
for pres in precision:
    mlsstd.append(np.mean(pres[msk] * 1e6))
mlsstd = np.array(mlsstd)
mls_pos = mlsmean_smooth + mlsstd
mls_neg = mlsmean_smooth - mlsstd

fig.add_trace(
    go.Scatter(
        x=m2v4dt,
        y=m2v4_pos,
        showlegend=False,
        line=dict(width=0),
        legendgroup="uncert",
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=m2v4dt,
        y=m2v4_neg,
        showlegend=False,
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(0,0,0,0.2)",
        name=r"MIRA2 uncertainty",
        legendgroup="uncert",
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=m2v4dt,
        y=m2v4mean,
        mode="markers",
        marker=dict(color="black", size=3),
        name="MIRA2",
        legendgroup="MIRA2",
        showlegend=False,
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=mlsdt,
        y=mlsmean_smooth * 1e6,
        mode="lines",
        line=dict(color="red"),
        name="MLS (convolved)",
        legendgroup="MLS",
        showlegend=False,
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=ed.date,
        y=np.full_like(m2v4_pos, max(m2v4_pos) + 0.5),
        mode="markers",
        marker=dict(color="black", symbol="circle-open"),
        name="PV EDGE (475K)",
        legendgroup="edge",
        showlegend=False,
    ),
    row=2,
    col=1,
)
fig.write_html(f"compare_fixed_and_changeing_apriori_{PMAX / 100}-{PMIN / 100}.html")
