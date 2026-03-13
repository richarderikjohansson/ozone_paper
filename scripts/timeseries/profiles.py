import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import date
from haversine import haversine
from cmcrameri import cm

cmap = cm.vik
vik = cmap.colors
selector = np.floor(np.linspace(0, 255, 41))

colors = []
for i in selector:
    color = vik[int(i), :]
    red = np.floor(color[0] * 255)
    green = np.floor(color[1] * 255)
    blue = np.floor(color[2] * 255)
    colors.append(f"rgba({red}, {green}, {blue}, 1)")


def read_data(file) -> dict:
    if "pgrid" in file.name or "apriori" in file.name:
        data = np.load(file, allow_pickle=True)
        return data
    else:
        data = np.load(file, allow_pickle=True).item()
        return data


def get_dataset(data, date):
    for dt, vals in data.items():
        data_date = dt.date()
        if data_date == date:
            return dt, vals


cwd = Path(__file__).parent / "data"
mls_path = cwd / "smooth/mls_smooth.npy"
m2_path = cwd / "matching/mira2_matching.npy"
apriori_path = cwd / "matching/apriori.npy"
pgrid_path = cwd / "matching/pgrid.npy"

mls = read_data(mls_path)
m2 = read_data(m2_path)
apriori = read_data(apriori_path)
pgrid = read_data(pgrid_path) / 1e2

over = date(2020, 3, 23)
good = date(2019, 11, 23)
m2overdt, m2over = get_dataset(data=m2, date=over)
mlsoverdt, mlsover = get_dataset(data=mls, date=over)

m2gooddt, m2good = get_dataset(data=m2, date=good)
mlsgooddt, mlsgood = get_dataset(data=mls, date=good)

apriori = m2good["apriori"] * 1e6
m2_good_ozone = apriori * m2good["x"]
ss_good = m2good["ss"]
eo_good = m2good["eo"]
err_good = ss_good + eo_good
mls_good_ozone = mlsgood["O3smooth"] * 1e6


mls_good_coords = [
    haversine((67.84, 20.41), (mlsgood["lat"], mlsgood["lon"])),
    mlsgood["lat"],
    mlsgood["lon"],
]

mls_over_coords = [
    haversine((67.84, 20.41), (mlsover["lat"], mlsover["lon"])),
    mlsover["lat"],
    mlsover["lon"],
]

m2_over_ozone = apriori * m2over["x"]
ss_over = m2over["ss"]
eo_over = m2over["eo"]
err_bad = ss_over + eo_over
mls_over_ozone = mlsover["O3smooth"] * 1e6


def plot_profiles():
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            f"MIRA2: {m2gooddt}, MLS: {mlsgooddt}",
            f"MIRA2: {m2overdt}, MLS: {mlsoverdt}",
        ],
    )
    fig.add_trace(
        go.Scatter(
            x=m2_good_ozone,
            y=pgrid,
            legendgroup="MIRA2",
            name="MIRA2",
            line=dict(color="black"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=m2_good_ozone - err_good,
            y=pgrid,
            fill="tonextx",
            line=dict(color="lightgray"),
            fillcolor="lightgray",
            legendgroup="MIRA2",
            name="MIRA2 error",
            showlegend=False,
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=m2_good_ozone + err_good,
            y=pgrid,
            fill="tonextx",
            line=dict(color="lightgray"),
            fillcolor="lightgray",
            legendgroup="MIRA2",
            name="MIRA2 error",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=m2_good_ozone,
            y=pgrid,
            legendgroup="MIRA2",
            name="MIRA2",
            line=dict(color="black"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=mls_good_ozone,
            y=pgrid,
            name="MLS",
            legendgroup="MLS",
            text=f"{mls_good_coords[0]}, {mls_good_coords[1]}, {mls_good_coords[2]}",
            line=dict(color="red"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=apriori,
            y=pgrid,
            name="MIRA2 apriori",
            legendgroup="MIRA2 apriori",
            line=dict(color="green", dash="dash"),
        ),
        row=1,
        col=1,
    )

    # over
    fig.add_trace(
        go.Scatter(
            x=m2_over_ozone,
            y=pgrid,
            legendgroup="MIRA2",
            line=dict(color="black"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=m2_over_ozone - err_bad,
            y=pgrid,
            fill="tonextx",
            line=dict(color="lightgray"),
            fillcolor="lightgray",
            legendgroup="MIRA2",
            name="MIRA2 error",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=m2_over_ozone + err_bad,
            y=pgrid,
            fill="tonextx",
            line=dict(color="lightgray"),
            fillcolor="lightgray",
            legendgroup="MIRA2",
            name="MIRA2 error",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=m2_over_ozone,
            y=pgrid,
            legendgroup="MIRA2",
            name="MIRA2",
            line=dict(color="black"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=mls_over_ozone,
            y=pgrid,
            name="MLS",
            legendgroup="MLS",
            line=dict(color="red"),
            text=f"{mls_over_coords[0]}, {mls_over_coords[1]}, {mls_over_coords[2]}",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=apriori,
            y=pgrid,
            name="MIRA2 apriori",
            showlegend=False,
            line=dict(color="green", dash="dash"),
        ),
        row=1,
        col=2,
    )

    fig.update_yaxes(
        autorange="reversed",
        visible=True,
        linecolor="black",
        linewidth=2,
        type="log",
        exponentformat="power",
        row=1,
    )
    fig.update_xaxes(
        visible=True,
        linecolor="black",
        linewidth=2,
        title="VMR [ppm]",
        range=[-1, 10.5],
        row=1,
    )
    fig.update_yaxes(title="Pressure [hPa]", row=1, col=1)
    fig.update_layout(width=1200, height=1000, plot_bgcolor="white")
    fig.write_html("profiles.html")


def plot_cfavks(data, pgrid, name):
    avks = data["avk"]
    mr = data["mr"]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["AVK's", "AVK's contours"],
        column_widths=[200, 1000],
        row_heights=[1000],
    )
    fig.update_layout(template="simple_white")
    fig.update_yaxes(row=1, col=1, type="log", autorange="reversed")
    fig.update_yaxes(row=1, col=2, type="log", autorange="reversed")
    fig.update_xaxes(row=1, col=2, type="log", autorange="reversed")
    for row, p, c in zip(avks, pgrid, colors):
        fig.add_trace(
            go.Scatter(
                x=row * 4,
                y=pgrid,
                mode="lines",
                name=f"p:{p:.2f}",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=mr,
            y=pgrid,
            name="MR",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(range=[-2, 2], row=1, col=1)
    fig.add_trace(go.Contour(z=avks.transpose(), x=pgrid, y=pgrid), row=1, col=2)
    fig.update_layout(
        width=1200,
        height=1000,
    )
    fig.write_html(name)


plot_cfavks(data=m2good, pgrid=pgrid, name="good_avk.html")
plot_cfavks(data=m2over, pgrid=pgrid, name="over_avk.html")
# plot_profiles()
