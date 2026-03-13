from h5py import File
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


from ozone.io import get_downloadsdir, get_egdefiles
from ozone.analysis import (
    get_period,
    propagate_uncertainty_mira2,
    make_weighted_mean,
    propagate_uncertainty_mls,
)
from ozone.utils import parse_edgefile, filter_edgedata

import matplotlib.pyplot as plt


mira2 = np.load(
    get_downloadsdir() / "MIRA2_v_1_1_screened_matching.npy",
    allow_pickle=True,
).item()

mls = np.load(
    get_downloadsdir() / "MLS_screened_matching.npy",
    allow_pickle=True,
).item()

m2day = get_period(mira2, period="day")
mlsday = get_period(mls, period="day")
efiles = get_egdefiles(root="/home/ric/Data/edge/")
edgedata = parse_edgefile(efiles[1])
edge = filter_edgedata(edgedata, 1)
edge_date = edge.date

covar = []
m2prod = []
m2pres = []
m2dt = [key for key in m2day.keys()]

for v in m2day.values():
    cov_ss = v["cov_ss"]
    cov_so = v["cov_so"]
    covar.append(cov_ss + cov_so)
    vmr = v["apriori"] * 1e6
    prod = v["x"]
    m2prod.append(vmr * prod)
    m2pres.append(v["pgrid"])

mlsprod = []
mlsdt = []
mlspres = []
mlsprecision = []
for dt, data in mls.items():
    print(data.keys())
    prod = data["O3_interp_smooth"] * 1e6
    mlsprod.append(prod)
    mlspres.append(data["p_interp"])
    mlsprecision.append(data["precision_interp"])

    mlsdt.append(dt)

m2means_bottom = [
    make_weighted_mean(prod, pres, 8, 12) for prod, pres in zip(m2prod, m2pres)
]

mlsmeans_bottom = [
    make_weighted_mean(prod, pres, 8, 12) for prod, pres in zip(mlsprod, mlspres)
]

uncert_mira2 = [
    propagate_uncertainty_mira2(cov, p, 8, 12) for cov, p in zip(covar, m2pres)
]

uncert_mls = [
    propagate_uncertainty_mls(precision, p, 8, 12)
    for precision, p in zip(mlsprecision, mlspres)
]

upper_mira2 = np.array(m2means_bottom) + np.array(uncert_mira2)
bottom_mira2 = np.array(m2means_bottom) - np.array(uncert_mira2)
fig = make_subplots(
    rows=1,
    cols=1,
    shared_xaxes=True,
    subplot_titles=[
        "30 < p < 100 hPa (~ 16 - 24 km)",
    ],
    x_title="Date [UTC]",
    y_title="Day O3 [VMR]",
)

fig.update_layout(template="plotly_white", height=350, width=1600)
fig.add_trace(go.Scatter(line=dict(width=0), x=m2dt, y=upper_mira2, showlegend=False))

fig.add_trace(
    go.Scatter(
        x=m2dt,
        y=bottom_mira2,
        fill="tonexty",
        fillcolor="rgba(0,100,80,0.2)",
        line=dict(width=0),
        name="MIRA 2 Confidence Interval",
    )
)
fig.add_trace(
    go.Scatter(
        x=m2dt,
        y=m2means_bottom,
        mode="lines",
        name="MIRA2",
        marker=dict(color="blue", size=5),
        legendgroup="MIRA2",
        showlegend=True,
    ),
    row=1,
    col=1,
)
mls_upper = np.array(mlsmeans_bottom) + np.mean(uncert_mls)
mls_bottom = np.array(mlsmeans_bottom) - np.mean(uncert_mls)

fig.add_trace(go.Scatter(line=dict(width=0), x=mlsdt, y=mls_upper, showlegend=False))

fig.add_trace(
    go.Scatter(
        x=mlsdt,
        y=mls_bottom,
        fill="tonexty",
        fillcolor="rgba(0,100,80,0.2)",
        line=dict(width=0),
        name="Confidence Interval",
    )
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

fig.write_html("100-30hPa.html")

# mlsmeans_bottom = [make_weighted_mean(x, mls_dp, 8, 12) for x in mlsprod]
#
# m2means_middle = [make_weighted_mean(x, m2_dp, 13, 18) for x in m2prod]
# mlsmeans_middle = [make_weighted_mean(x, mls_dp, 13, 18) for x in mlsprod]
#
# m2means_top = [make_weighted_mean(x, m2_dp, 19, 23) for x in m2prod]
# mlsmeans_top = [make_weighted_mean(x, mls_dp, 19, 23) for x in mlsprod]
