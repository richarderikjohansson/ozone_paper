import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt


from ozone.io import get_downloadsdir
from ozone.analysis import get_period, make_weighted_mean, propagate_uncertainty_mira2


def get_data(period):
    m2file = get_downloadsdir() / "MIRA2_v_3_screened_matching.npy"
    mlsfile = get_downloadsdir() / "MLS_screened_matching.npy"

    mira2 = np.load(m2file, allow_pickle=True).item()
    mls = np.load(mlsfile, allow_pickle=True).item()

    m2 = get_period(mira2, period=period)
    mls = get_period(mls, period=period)
    return m2, mls


def smooth_mira2(avk, vmr, apriori):
    shape = avk.shape[0]
    identity = np.identity(n=shape)
    smoothed = (identity - avk) @ apriori + avk @ vmr
    return smoothed


m2, mls = get_data("day")
m2_pres = np.array([v["pgrid"] / 1e2 for v in m2.values()]).mean(axis=0)
m2_alt = np.array([v["zgrid"] for v in m2.values()]).mean(axis=0)
m2_sig = np.array([v["ss"] + v["eo"] for v in m2.values()])
mls_sig = np.array([v["precision_interp"] for v in mls.values()])
mls_pres = np.array([v["p_interp"] / 1e2 for v in mls.values()]).mean(axis=0)

m2dt = np.array([dt for dt in m2.keys()])
mlsdt = np.array([dt for dt in mls.keys()])

m2vmr = []
mlsvmr = []

for v in m2.values():
    apriori = v["apriori"] * 1e6
    avk = v["avk"]
    vmr = apriori * v["x"]
    m2vmr.append(smooth_mira2(avk, vmr, apriori))


for v in mls.values():
    prod = v["O3_interp_smooth"] * 1e6
    mlsvmr.append(prod)


m2mean_1 = np.array([make_weighted_mean(prod, m2_pres, 8, 13) for prod in m2vmr])
mlsmean_1 = np.array([make_weighted_mean(prod, mls_pres, 8, 13) for prod in mlsvmr])
m2_uncert_1 = np.array([make_weighted_mean(sig, m2_pres, 8, 13) for sig in m2_sig])

m2mean_2 = np.array([make_weighted_mean(prod, m2_pres, 13, 18) for prod in m2vmr])
mlsmean_2 = np.array([make_weighted_mean(prod, mls_pres, 13, 18) for prod in mlsvmr])
m2_uncert_2 = np.array([make_weighted_mean(sig, m2_pres, 13, 18) for sig in m2_sig])

m2mean_3 = np.array([make_weighted_mean(prod, m2_pres, 18, 24) for prod in m2vmr])
mlsmean_3 = np.array([make_weighted_mean(prod, mls_pres, 18, 24) for prod in mlsvmr])
m2_uncert_3 = np.array([make_weighted_mean(sig, m2_pres, 18, 24) for sig in m2_sig])

m2mean_4 = np.array([make_weighted_mean(prod, m2_pres, 24, 32) for prod in m2vmr])
mlsmean_4 = np.array([make_weighted_mean(prod, mls_pres, 24, 32) for prod in mlsvmr])
m2_uncert_4 = np.array([make_weighted_mean(sig, m2_pres, 24, 32) for sig in m2_sig])


fig = make_subplots(
    4,
    1,
    shared_xaxes=True,
    row_titles=["16-25 km", "25-34 km", "34-46 km", "46-63 km"],
)
fig.update_layout(template="plotly_white", height=1000, width=1400)
fig.add_trace(
    go.Scatter(
        x=m2dt,
        y=m2mean_1 + m2_uncert_1,
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
        y=m2mean_1 - m2_uncert_1,
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
        y=m2mean_1,
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
        y=mlsmean_1,
        mode="lines",
        line=dict(color="red"),
        name="MLS",
        legendgroup="MLS",
        showlegend=True,
    ),
    row=1,
    col=1,
)
# row 2
fig.add_trace(
    go.Scatter(
        x=m2dt,
        y=m2mean_2 + m2_uncert_2,
        showlegend=False,
        line=dict(width=0),
        legendgroup="uncert",
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=m2dt,
        y=m2mean_2 - m2_uncert_2,
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
        x=m2dt,
        y=m2mean_2,
        mode="markers",
        marker=dict(color="black", size=3),
        name="MIRA2",
        showlegend=False,
        legendgroup="MIRA2",
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=mlsdt,
        y=mlsmean_2,
        mode="lines",
        line=dict(color="red"),
        name="MLS",
        legendgroup="MLS",
        showlegend=False,
    ),
    row=2,
    col=1,
)
# row 3
fig.add_trace(
    go.Scatter(
        x=m2dt,
        y=m2mean_3 + m2_uncert_3,
        showlegend=False,
        line=dict(width=0),
        legendgroup="uncert",
    ),
    row=3,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=m2dt,
        y=m2mean_3 - m2_uncert_3,
        showlegend=False,
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(0,0,0,0.2)",
        name=r"$\pm \sigma$",
        legendgroup="uncert",
    ),
    row=3,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=m2dt,
        y=m2mean_3,
        mode="markers",
        marker=dict(color="black", size=3),
        name="MIRA2",
        legendgroup="MIRA2",
        showlegend=False,
    ),
    row=3,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=mlsdt,
        y=mlsmean_3,
        mode="lines",
        line=dict(color="red"),
        name="MLS",
        legendgroup="MLS",
        showlegend=False,
    ),
    row=3,
    col=1,
)
# row 4
fig.add_trace(
    go.Scatter(
        x=m2dt,
        y=m2mean_4 + m2_uncert_4,
        showlegend=False,
        line=dict(width=0),
        legendgroup="uncert",
    ),
    row=4,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=m2dt,
        y=m2mean_4 - m2_uncert_4,
        showlegend=False,
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(0,0,0,0.2)",
        name=r"$\pm \sigma$",
        legendgroup="uncert",
    ),
    row=4,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=m2dt,
        y=m2mean_4,
        mode="markers",
        marker=dict(color="black", size=3),
        name="MIRA2",
        legendgroup="MIRA2",
        showlegend=False,
    ),
    row=4,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=mlsdt,
        y=mlsmean_4,
        mode="lines",
        line=dict(color="red"),
        name="MLS",
        legendgroup="MLS",
        showlegend=False,
    ),
    row=4,
    col=1,
)

fig.write_html("mira2_v3_day.html")

m2, mls = get_data("night")
m2_pres = np.array([v["pgrid"] / 1e2 for v in m2.values()]).mean(axis=0)
m2_alt = np.array([v["zgrid"] for v in m2.values()]).mean(axis=0)
m2_sig = np.array([v["ss"] + v["eo"] for v in m2.values()])
mls_sig = np.array([v["precision_interp"] for v in mls.values()])
mls_pres = np.array([v["p_interp"] / 1e2 for v in mls.values()]).mean(axis=0)

m2dt = np.array([dt for dt in m2.keys()])
mlsdt = np.array([dt for dt in mls.keys()])

m2vmr = []
mlsvmr = []

for v in m2.values():
    apriori = v["apriori"] * 1e6
    avk = v["avk"]
    vmr = apriori * v["x"]
    m2vmr.append(smooth_mira2(avk, vmr, apriori))


for v in mls.values():
    prod = v["O3_interp_smooth"] * 1e6
    mlsvmr.append(prod)


m2mean_1 = np.array([make_weighted_mean(prod, m2_pres, 8, 13) for prod in m2vmr])
mlsmean_1 = np.array([make_weighted_mean(prod, mls_pres, 8, 13) for prod in mlsvmr])
m2_uncert_1 = np.array([make_weighted_mean(sig, m2_pres, 8, 13) for sig in m2_sig])

m2mean_2 = np.array([make_weighted_mean(prod, m2_pres, 13, 18) for prod in m2vmr])
mlsmean_2 = np.array([make_weighted_mean(prod, mls_pres, 13, 18) for prod in mlsvmr])
m2_uncert_2 = np.array([make_weighted_mean(sig, m2_pres, 13, 18) for sig in m2_sig])

m2mean_3 = np.array([make_weighted_mean(prod, m2_pres, 18, 24) for prod in m2vmr])
mlsmean_3 = np.array([make_weighted_mean(prod, mls_pres, 18, 24) for prod in mlsvmr])
m2_uncert_3 = np.array([make_weighted_mean(sig, m2_pres, 18, 24) for sig in m2_sig])

m2mean_4 = np.array([make_weighted_mean(prod, m2_pres, 24, 32) for prod in m2vmr])
mlsmean_4 = np.array([make_weighted_mean(prod, mls_pres, 24, 32) for prod in mlsvmr])
m2_uncert_4 = np.array([make_weighted_mean(sig, m2_pres, 24, 32) for sig in m2_sig])

fig = make_subplots(
    4,
    1,
    shared_xaxes=True,
    row_titles=["16-25 km", "25-34 km", "34-46 km", "46-63 km"],
)
fig.update_layout(template="plotly_white", height=1000, width=1400)
fig.add_trace(
    go.Scatter(
        x=m2dt,
        y=m2mean_1 + m2_uncert_1,
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
        y=m2mean_1 - m2_uncert_1,
        showlegend=True,
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(0,0,0,0.2)",
        name=r"$\pm \sigma$",
        legendgroup="uncert",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=m2dt,
        y=m2mean_1,
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
        y=mlsmean_1,
        mode="lines",
        line=dict(color="red"),
        name="MLS",
        legendgroup="MLS",
        showlegend=True,
    ),
    row=1,
    col=1,
)
# row 2
fig.add_trace(
    go.Scatter(
        x=m2dt,
        y=m2mean_2 + m2_uncert_2,
        showlegend=False,
        line=dict(width=0),
        legendgroup="uncert",
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=m2dt,
        y=m2mean_2 - m2_uncert_2,
        showlegend=False,
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(0,0,0,0.2)",
        name=r"$\pm \sigma$",
        legendgroup="uncert",
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=m2dt,
        y=m2mean_2,
        mode="markers",
        marker=dict(color="black", size=3),
        name="MIRA2",
        showlegend=False,
        legendgroup="MIRA2",
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=mlsdt,
        y=mlsmean_2,
        mode="lines",
        line=dict(color="red"),
        name="MLS",
        legendgroup="MLS",
        showlegend=False,
    ),
    row=2,
    col=1,
)
# row 3
fig.add_trace(
    go.Scatter(
        x=m2dt,
        y=m2mean_3 + m2_uncert_3,
        showlegend=False,
        line=dict(width=0),
        legendgroup="uncert",
    ),
    row=3,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=m2dt,
        y=m2mean_3 - m2_uncert_3,
        showlegend=False,
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(0,0,0,0.2)",
        name=r"$\pm \sigma$",
        legendgroup="uncert",
    ),
    row=3,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=m2dt,
        y=m2mean_3,
        mode="markers",
        marker=dict(color="black", size=3),
        name="MIRA2",
        legendgroup="MIRA2",
        showlegend=False,
    ),
    row=3,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=mlsdt,
        y=mlsmean_3,
        mode="lines",
        line=dict(color="red"),
        name="MLS",
        legendgroup="MLS",
        showlegend=False,
    ),
    row=3,
    col=1,
)
# row 4
fig.add_trace(
    go.Scatter(
        x=m2dt,
        y=m2mean_4 + m2_uncert_4,
        showlegend=False,
        line=dict(width=0),
        legendgroup="uncert",
    ),
    row=4,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=m2dt,
        y=m2mean_4 - m2_uncert_4,
        showlegend=False,
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(0,0,0,0.2)",
        name=r"$\pm \sigma$",
        legendgroup="uncert",
    ),
    row=4,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=m2dt,
        y=m2mean_4,
        mode="markers",
        marker=dict(color="black", size=3),
        name="MIRA2",
        legendgroup="MIRA2",
        showlegend=False,
    ),
    row=4,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=mlsdt,
        y=mlsmean_4,
        mode="lines",
        line=dict(color="red"),
        name="MLS",
        legendgroup="MLS",
        showlegend=False,
    ),
    row=4,
    col=1,
)

fig.write_html("mira2_v3_night.html")
