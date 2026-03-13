import numpy as np
from ozone.analysis import pressure_thickness
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ozone.io import get_egdefiles
from ozone.utils import parse_edgefile, filter_edgedata


def combine_data():
    m2_n = np.load("mira2_matching_night.npy", allow_pickle=True).item()
    m2_d = np.load("mira2_matching.npy", allow_pickle=True).item()

    mls_n = np.load("mls_matching_night.npy", allow_pickle=True).item()
    mls_d = np.load("mls_matching.npy", allow_pickle=True).item()

    m2_n_dt = np.array([k for k in m2_n.keys()])
    m2_d_dt = np.array([k for k in m2_d.keys()])
    m2dt = sorted(np.concatenate([m2_d_dt, m2_n_dt]))

    mls_n_dt = np.array([k for k in mls_n.keys()])
    mls_d_dt = np.array([k for k in mls_d.keys()])
    mlsdt = sorted(np.concatenate([mls_d_dt, mls_n_dt]))

    m2 = dict()
    for dt in m2dt:
        try:
            m2[dt] = m2_n[dt]

        except KeyError:
            m2[dt] = m2_d[dt]

    mls = dict()
    for dt in mlsdt:
        try:
            mls[dt] = mls_n[dt]

        except KeyError:
            mls[dt] = mls_d[dt]

    return m2, mls


def make_weighted_mean(x, dp, s, e):
    w = np.zeros(len(x))
    w[s:e] = dp[s:e] / np.sum(dp[s:e])
    x_mean = w @ x
    return x_mean


def make_mean(x, s, e):
    xsub = x[s:e]
    x_mean = xsub.mean()
    return x_mean


# m2, mls = combine_data()
m2 = np.load("mira2_matching.npy", allow_pickle=True).item()
mls = np.load("mls_matching.npy", allow_pickle=True).item()

m2prod = []
m2dt = []
for dt, data in m2.items():
    apriori = data["apriori"]
    x = data["x"]
    prod = apriori * x * 1e6
    m2prod.append(prod)
    m2dt.append(dt)
    m2pres = data["pgrid"] / 1e2

mlsprod = []
mlsdt = []
for dt, data in mls.items():
    prod = data["O3smooth"] * 1e6
    mlsprod.append(prod)
    mlspres = data["pnew"]
    mlsdt.append(dt)


m2prod = np.array(m2prod)
mlsprod = np.array(mlsprod)
mls_dp = pressure_thickness(mlspres)
m2_dp = pressure_thickness(m2pres)
efiles = get_egdefiles(root="/home/ric/Data/edge/")
edgedata = parse_edgefile(efiles[1])
edge = filter_edgedata(edgedata, 1)
edge_date = edge.date


m2means_bottom = [make_weighted_mean(x, m2_dp, 8, 12) for x in m2prod]
mlsmeans_bottom = [make_weighted_mean(x, mls_dp, 8, 12) for x in mlsprod]

m2means_middle = [make_weighted_mean(x, m2_dp, 13, 18) for x in m2prod]
mlsmeans_middle = [make_weighted_mean(x, mls_dp, 13, 18) for x in mlsprod]

m2means_top = [make_weighted_mean(x, m2_dp, 19, 23) for x in m2prod]
mlsmeans_top = [make_weighted_mean(x, mls_dp, 19, 23) for x in mlsprod]

for v in m2.values():
    alt = v["zgrid"]

for i, p in enumerate(alt):
    print(i, p)


fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    subplot_titles=[
        "30 < p < 100 hPa (~ 16 - 24 km)",
        "5 < p < 25 hPa   (~ 25 - 35 km)",
        "1 < p < 5 hPa    (~ 37 - 46 km)",
    ],
    x_title="Date [UTC]",
    y_title="Day O3 [VMR]",
)

fig.update_layout(template="plotly_white", height=1000, width=1600)
fig.add_trace(
    go.Scatter(
        x=m2dt,
        y=m2means_bottom,
        mode="markers",
        name="MIRA2",
        marker=dict(color="blue", size=5),
        legendgroup="MIRA2",
        showlegend=True,
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=edge_date,
        y=np.full_like(edge_date, 2.7),
        mode="markers",
        name="Inside PV (475 K)",
        marker=dict(color="black", size=5),
        legendgroup="EDGE",
        showlegend=True,
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=mlsdt,
        y=mlsmeans_bottom,
        name="MLS convolved",
        line=dict(color="red", width=3),
        legendgroup="MLS",
        showlegend=True,
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=m2dt,
        y=m2means_middle,
        mode="markers",
        marker=dict(color="blue", size=5),
        legendgroup="MIRA2",
        showlegend=False,
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=edge_date,
        y=np.full_like(edge_date, 8.7),
        mode="markers",
        name="Inside PV",
        marker=dict(color="black", size=5),
        legendgroup="EDGE",
        showlegend=False,
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=mlsdt,
        y=mlsmeans_middle,
        line=dict(color="red", width=3),
        legendgroup="MLS",
        showlegend=False,
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=m2dt,
        y=m2means_top,
        mode="markers",
        marker=dict(color="blue", size=5),
        legendgroup="MIRA2",
        showlegend=False,
    ),
    row=3,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=edge_date,
        y=np.full_like(edge_date, 8.9),
        mode="markers",
        name="Inside PV",
        marker=dict(color="black", size=5),
        legendgroup="EDGE",
        showlegend=False,
    ),
    row=3,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=mlsdt,
        y=mlsmeans_top,
        line=dict(color="red", width=3),
        legendgroup="MLS",
        showlegend=False,
    ),
    row=3,
    col=1,
)
fig.write_html("vmr_means_day_v1.html")

m2 = np.load("mira2_matching_night.npy", allow_pickle=True).item()
mls = np.load("mls_matching_night.npy", allow_pickle=True).item()

m2prod = []
m2dt = []
for dt, data in m2.items():
    apriori = data["apriori"]
    x = data["x"]
    prod = apriori * x * 1e6
    m2prod.append(prod)
    m2dt.append(dt)
    m2pres = data["pgrid"] / 1e2

mlsprod = []
mlsdt = []
for dt, data in mls.items():
    prod = data["O3smooth"] * 1e6
    mlsprod.append(prod)
    mlspres = data["pnew"]
    mlsdt.append(dt)


m2prod = np.array(m2prod)
mlsprod = np.array(mlsprod)
mls_dp = pressure_thickness(mlspres)
m2_dp = pressure_thickness(m2pres)
efiles = get_egdefiles(root="/home/ric/Data/edge/")
edgedata = parse_edgefile(efiles[1])
edge = filter_edgedata(edgedata, 1)
edge_date = edge.date


m2means_bottom = [make_weighted_mean(x, m2_dp, 8, 12) for x in m2prod]
mlsmeans_bottom = [make_weighted_mean(x, mls_dp, 8, 12) for x in mlsprod]

m2means_middle = [make_weighted_mean(x, m2_dp, 13, 18) for x in m2prod]
mlsmeans_middle = [make_weighted_mean(x, mls_dp, 13, 18) for x in mlsprod]

m2means_top = [make_weighted_mean(x, m2_dp, 19, 23) for x in m2prod]
mlsmeans_top = [make_weighted_mean(x, mls_dp, 19, 23) for x in mlsprod]

for i, p in enumerate(m2pres):
    print(i, p)


fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    subplot_titles=[
        "30 < p < 100 hPa (~ 16 - 24 km)",
        "5 < p < 25 hPa   (~ 25 - 35 km)",
        "1 < p < 5 hPa    (~ 37 - 46 km)",
    ],
    x_title="Date [UTC]",
    y_title="Night O3 [VMR]",
)

fig.update_layout(template="plotly_white", height=1000, width=1600)
fig.add_trace(
    go.Scatter(
        x=m2dt,
        y=m2means_bottom,
        mode="markers",
        name="MIRA2",
        marker=dict(color="blue", size=5),
        legendgroup="MIRA2",
        showlegend=True,
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=edge_date,
        y=np.full_like(edge_date, 2.7),
        mode="markers",
        name="Inside PV (475 K)",
        marker=dict(color="black", size=5),
        legendgroup="EDGE",
        showlegend=True,
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=mlsdt,
        y=mlsmeans_bottom,
        name="MLS convolved",
        line=dict(color="red", width=3),
        legendgroup="MLS",
        showlegend=True,
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=m2dt,
        y=m2means_middle,
        mode="markers",
        marker=dict(color="blue", size=5),
        legendgroup="MIRA2",
        showlegend=False,
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=edge_date,
        y=np.full_like(edge_date, 8.7),
        mode="markers",
        name="Inside PV",
        marker=dict(color="black", size=5),
        legendgroup="EDGE",
        showlegend=False,
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=mlsdt,
        y=mlsmeans_middle,
        line=dict(color="red", width=3),
        legendgroup="MLS",
        showlegend=False,
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=m2dt,
        y=m2means_top,
        mode="markers",
        marker=dict(color="blue", size=5),
        legendgroup="MIRA2",
        showlegend=False,
    ),
    row=3,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=edge_date,
        y=np.full_like(edge_date, 8.9),
        mode="markers",
        name="Inside PV",
        marker=dict(color="black", size=5),
        legendgroup="EDGE",
        showlegend=False,
    ),
    row=3,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=mlsdt,
        y=mlsmeans_top,
        line=dict(color="red", width=3),
        legendgroup="MLS",
        showlegend=False,
    ),
    row=3,
    col=1,
)
fig.write_html("vmr_means_night_v1.html")
