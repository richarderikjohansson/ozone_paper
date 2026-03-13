import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import plotly.graph_objects as go
from dataclasses import dataclass, fields
from numpy.typing import NDArray
from haversine import haversine, Unit


@dataclass(slots=True)
class M2Struct:
    ozone: list
    apriori: list
    dt: list
    ss: list
    eo: list
    pressure: list


@dataclass(slots=True)
class MLSstruct:
    ozone: list
    dt: list
    lat: list
    lon: list
    pressure: list
    distance: list


@dataclass(slots=True)
class Data7HPa:
    m2: list
    m2dt: list
    m2err: list
    m2apriori: list

    mls: list
    mlsdt: list
    mlsdt: list
    mlslon: list
    mlslat: list


@dataclass(slots=True)
class Data4HPa:
    m2: list
    m2dt: list
    m2err: list
    m2apriori: list

    mls: list
    mlsdt: list
    mlsdt: list
    mlslon: list
    mlslat: list


@dataclass(slots=True)
class Data17HPa:
    m2: list
    m2dt: list
    m2err: list
    m2apriori: list

    mls: list
    mlsdt: list
    mlslon: list
    mlslat: list


def find_files(dataset) -> Path:
    cwd = Path(__file__).parent
    datadir = cwd / "data"

    match dataset:
        case "mira2":
            d = datadir / "matching_night"
            data = d / "mira2_matching_night.npy"
        case "mls":
            d = datadir / "smooth"
            data = d / "mls_smooth_night.npy"
        case "pgrid":
            d = datadir / "matching_night"
            data = d / "pgrid.npy"

    return data


def read_data(file) -> dict:
    if "pgrid" in file.name or "apriori" in file.name:
        data = np.load(file, allow_pickle=True)
        return data
    else:
        data = np.load(file, allow_pickle=True).item()
        return data


def get_pressure_level_index(p_hpa):
    for pl in p_hpa:
        if 1 <= pl <= 5:
            i = pl
        if 7 <= pl <= 9:
            j = pl
        if 10 <= pl <= 20:
            k = pl

    i4 = np.where(p_hpa == i)[0][0]
    i7 = np.where(p_hpa == j)[0][0]
    i17 = np.where(p_hpa == k)[0][0]

    return i4, i7, i17


def set_structs(mls: dict, mira2: dict, p_hpa: NDArray):
    latitude = [val["lat"] for val in mls.values()]
    longitude = [val["lon"] for val in mls.values()]
    M2Struct.ozone = [(val["apriori"] * val["x"]) * 1e6 for val in mira2.values()]
    M2Struct.apriori = [val["apriori"] * 1e6 for val in mira2.values()]
    M2Struct.dt = [dt for dt in mira2.keys()]
    M2Struct.ss = [val["ss"] for val in mira2.values()]
    M2Struct.eo = [val["eo"] for val in mira2.values()]

    MLSstruct.ozone = [val["O3smooth"] * 1e6 for val in mls.values()]
    MLSstruct.dt = [dt for dt in mls.keys()]
    MLSstruct.lat = latitude
    MLSstruct.lon = longitude
    MLSstruct.distance = [
        haversine((67.84, 20.41), (lat, lon)) for lat, lon in zip(latitude, longitude)
    ]


def get_linedata(i4, i7, i17):
    Data4HPa.m2 = [arr[i4] for arr in M2Struct.ozone]
    Data4HPa.m2dt = M2Struct.dt
    Data4HPa.m2apriori = [arr[i4] for arr in M2Struct.apriori]
    Data4HPa.m2err = [ss[i4] + eo[i4] for ss, eo in zip(M2Struct.ss, M2Struct.eo)]
    Data4HPa.mls = [arr[i4] for arr in MLSstruct.ozone]
    Data4HPa.mlsdt = MLSstruct.dt
    Data4HPa.mlslon = MLSstruct.lon
    Data4HPa.mlslat = MLSstruct.lat

    Data7HPa.m2 = [arr[i7] for arr in M2Struct.ozone]
    Data7HPa.m2dt = M2Struct.dt
    Data7HPa.m2apriori = [arr[i7] for arr in M2Struct.apriori]
    Data7HPa.m2err = [ss[i7] + eo[i7] for ss, eo in zip(M2Struct.ss, M2Struct.eo)]
    Data7HPa.mls = [arr[i7] for arr in MLSstruct.ozone]
    Data7HPa.mlsdt = MLSstruct.dt
    Data7HPa.mlslon = MLSstruct.lon
    Data7HPa.mlslat = MLSstruct.lat

    Data17HPa.m2 = [arr[i17] for arr in M2Struct.ozone]
    Data17HPa.m2dt = M2Struct.dt
    Data17HPa.m2apriori = [arr[i17] for arr in M2Struct.apriori]
    Data17HPa.m2err = [ss[i17] + eo[i17] for ss, eo in zip(M2Struct.ss, M2Struct.eo)]
    Data17HPa.mls = [arr[i17] for arr in MLSstruct.ozone]
    Data17HPa.mlsdt = MLSstruct.dt
    Data17HPa.mlslon = MLSstruct.lon
    Data17HPa.mlslat = MLSstruct.lat


def plot_line(mls, mira2, p_hpa):
    set_structs(mls, mira2, p_hpa)
    i4, i7, i17 = get_pressure_level_index(p_hpa)
    get_linedata(i4, i7, i17)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=Data4HPa.m2dt,
            y=Data4HPa.m2,
            mode="markers",
            error_y=dict(type="data", array=Data4HPa.m2err, visible=True),
            name="MIRA2",
            line=dict(color="black"),
            marker=dict(color="black"),
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=Data4HPa.m2dt,
            y=Data4HPa.m2apriori,
            line=dict(color="green"),
            name="MIRA2 apriori",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=Data4HPa.mlsdt,
            y=Data4HPa.mls,
            text=[
                f"lon:{lon}, lat:{lat}, distance: {d}"
                for lon, lat, d in zip(
                    Data4HPa.mlslon, Data4HPa.mlslat, MLSstruct.distance
                )
            ],
            line=dict(color="red"),
            name="MLS",
        )
    )
    fig.update_layout(
        title=r"4 HPa pressure level",
        width=1400,
        height=600,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend={"x": 0.05, "y": 0.95, "xanchor": "left", "yanchor": "top"},
    )
    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        title_text="Date",
        tickfont=dict(size=16),
        title_font=dict(size=18),
    )
    fig.update_yaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        range=[0, 10],
        title_text="VMR [ppm]",
        tickfont=dict(size=16),
        title_font=dict(size=18),
    )
    fig.update_traces(showlegend=True)
    fig.write_html("4Hpa_night.html")
    del fig

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=Data7HPa.m2dt,
            y=Data7HPa.m2,
            mode="markers",
            error_y=dict(type="data", array=Data7HPa.m2err, visible=True),
            name="MIRA2",
            line=dict(color="black"),
            marker=dict(color="black"),
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=Data7HPa.m2dt,
            y=Data7HPa.m2apriori,
            line=dict(color="green"),
            name="MIRA2 apriori",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=Data7HPa.mlsdt,
            y=Data7HPa.mls,
            text=[
                f"lon:{lon}, lat:{lat}, distance: {d}"
                for lon, lat, d in zip(
                    Data7HPa.mlslon, Data7HPa.mlslat, MLSstruct.distance
                )
            ],
            line=dict(color="red"),
            name="MLS",
        )
    )
    fig.update_layout(
        title=r"7 HPa pressure level",
        width=1400,
        height=600,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        title_text="Date",
        tickfont=dict(size=16),
        title_font=dict(size=18),
    )
    fig.update_yaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        range=[0, 10],
        title_text="VMR [ppm]",
        tickfont=dict(size=16),
        title_font=dict(size=18),
    )
    fig.update_traces(showlegend=True)
    fig.write_html("7Hpa_night.html")
    del fig

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=Data17HPa.m2dt,
            y=Data17HPa.m2,
            mode="markers",
            error_y=dict(type="data", array=Data17HPa.m2err, visible=True),
            name="MIRA2",
            line=dict(color="black"),
            marker=dict(color="black"),
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=Data17HPa.m2dt,
            y=Data17HPa.m2apriori,
            line=dict(color="green"),
            name="MIRA2 apriori",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=Data17HPa.mlsdt,
            y=Data17HPa.mls,
            text=[
                f"lon:{lon}, lat:{lat}, distance: {d}"
                for lon, lat, d in zip(
                    Data17HPa.mlslon, Data17HPa.mlslat, MLSstruct.distance
                )
            ],
            line=dict(color="red"),
            name="MLS",
        )
    )
    fig.update_layout(
        title=r"17 HPa pressure level",
        width=1400,
        height=600,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        title_text="Date",
        tickfont=dict(size=16),
        title_font=dict(size=18),
    )
    fig.update_yaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        range=[0, 10],
        title_text="VMR [ppm]",
        tickfont=dict(size=16),
        title_font=dict(size=18),
    )
    fig.update_traces(showlegend=True)
    fig.write_html("17Hpa_night.html")
    del fig


def plot_line_matplotlib(mls, mira2, p_hpa):
    for pl in p_hpa:
        if 7 <= pl <= 9:
            break
    i_7hpa = np.where(p_hpa == pl)[0][0]
    m2O3 = [(val["apriori"] * val["x"]) * 1e6 for val in mira2.values()]
    ss = [val["ss"] for val in mira2.values()]
    eo = [val["eo"] for val in mira2.values()]
    m2dt = [dt for dt in mira2.keys()]

    mlsO3 = [val["O3smooth"] * 1e6 for val in mls.values()]
    mlsdt = [dt for dt in mls.keys()]

    m2O3_7hpa = []
    mlsO3_7hpa = []
    err = []

    for s, e, v in zip(ss, eo, m2O3):
        m2O3_7hpa.append(v[i_7hpa])
        err.append(s[i_7hpa] + e[i_7hpa])

    for v in mlsO3:
        mlsO3_7hpa.append(v[i_7hpa])

    fig = plt.figure(figsize=(15, 4))
    gs = GridSpec(1, 1)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title(r"MIRA and MLS VMR $7 \leq p \leq 9$ hPa", fontsize=18)
    ax.errorbar(
        x=m2dt,
        y=m2O3_7hpa,
        yerr=err,
        color="black",
        ls="",
        marker=".",
        capsize=3,
        label="MIRA2",
        markersize=2,
    )
    ax.scatter(x=mlsdt, y=mlsO3_7hpa, c="tomato", s=6, zorder=4, label="MLS (smoothed)")
    ax.set_ylabel(r"$O_3$ [ppm]", fontsize=14)
    ax.set_xlabel("Date", fontsize=14)
    ax.tick_params(labelsize=11)
    ax.legend(loc="upper left")
    plt.tight_layout()
    fig.savefig("m2_7hpa.pdf")

    for pl in p_hpa:
        if 1 <= pl <= 5:
            break

    m2O3_4hpa = []
    mlsO3_4hpa = []
    err = []
    i_4hpa = np.where(p_hpa == pl)[0][0]

    for s, e, v in zip(ss, eo, m2O3):
        m2O3_4hpa.append(v[i_4hpa])
        err.append(s[i_4hpa] + e[i_4hpa])

    for v in mlsO3:
        mlsO3_4hpa.append(v[i_4hpa])

    fig = plt.figure(figsize=(15, 4))
    gs = GridSpec(1, 1)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title(r"MIRA and MLS VMR $1 \leq p \leq 5$ hPa", fontsize=18)
    ax.errorbar(
        x=m2dt,
        y=m2O3_4hpa,
        yerr=err,
        color="black",
        ls="",
        marker=".",
        capsize=3,
        label="MIRA2",
        markersize=2,
    )
    ax.scatter(x=mlsdt, y=mlsO3_4hpa, c="tomato", s=6, zorder=4, label="MLS (smoothed)")
    ax.set_ylabel(r"$O_3$ [ppm]", fontsize=14)
    ax.set_xlabel("Date", fontsize=14)
    ax.tick_params(labelsize=11)
    ax.legend(loc="upper left")
    plt.tight_layout()
    fig.savefig("m2_4hpa.pdf")

    for pl in p_hpa:
        if 10 <= pl <= 20:
            break

    m2O3_17hpa = []
    mlsO3_17hpa = []
    err = []
    i_17hpa = np.where(p_hpa == pl)[0][0]

    for s, e, v in zip(ss, eo, m2O3):
        m2O3_17hpa.append(v[i_17hpa])
        err.append(s[i_17hpa] + e[i_17hpa])

    for v in mlsO3:
        mlsO3_17hpa.append(v[i_17hpa])

    fig = plt.figure(figsize=(15, 4))
    gs = GridSpec(1, 1)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title(r"MIRA and MLS VMR $10 \leq p \leq 20$ hPa", fontsize=18)
    ax.errorbar(
        x=m2dt,
        y=m2O3_17hpa,
        yerr=err,
        color="black",
        ls="",
        marker=".",
        capsize=3,
        label="MIRA2",
        markersize=2,
    )
    ax.scatter(
        x=mlsdt, y=mlsO3_17hpa, c="tomato", s=6, zorder=4, label="MLS (smoothed)"
    )
    ax.set_ylabel(r"$O_3$ [ppm]", fontsize=14)
    ax.set_xlabel("Date", fontsize=14)
    ax.tick_params(labelsize=11)
    ax.legend(loc="upper left")
    plt.tight_layout()
    fig.savefig("m2_17hpa.pdf")


mls = read_data(find_files(dataset="mls"))
mira2 = read_data(find_files(dataset="mira2"))
pgrid = read_data(find_files(dataset="pgrid"))
p_hpa = pgrid / 1e2

plot_line(mls, mira2, p_hpa)
