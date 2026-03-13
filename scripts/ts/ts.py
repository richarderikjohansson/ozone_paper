from pathlib import Path
from ozone.io import get_egdefiles, get_data_files_root
from ozone.utils import parse_edgefile, filter_edgedata, EdgeData, MIRA2Data, MLSData
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
import plotly.graph_objects as go
from datetime import date as datetime_date


@dataclass
class PressureLevelData:
    pl: float
    mira2_dt: NDArray
    mls_dt: NDArray
    mls: NDArray
    mls_lon: NDArray
    mls_lat: NDArray
    mira2: NDArray
    mira2_apriori: NDArray
    err: NDArray


def read_profiles(which: str) -> (MLSData, MIRA2Data):
    files, datadir = get_data_files_root(ext="npy")

    if which == "night":
        ddir = "matching_night"
    elif which == "day":
        ddir = "matching"

    mira2_parent_path = datadir / ddir
    mls_parent_path = datadir / "smooth"
    assert mira2_parent_path.exists()
    assert mls_parent_path.exists()

    match which:
        case "night":
            mls_files = ["mls_smooth_night.npy", "pgrid.npy"]
            mira2_files = ["mira2_matching_night.npy", "pgrid.npy", "apriori.npy"]
        case "day":
            mls_files = ["mls_smooth.npy", "pgrid.npy"]
            mira2_files = ["mira2_matching.npy", "pgrid.npy", "apriori.npy"]
        case _:
            print("Could not find files")

    mls_paths = [mls_parent_path / file for file in mls_files]
    mira2_paths = [mira2_parent_path / file for file in mira2_files]

    dt = []
    date = []
    time = []
    prod = []
    err = []
    for path in mira2_paths:
        name = path.name
        if "matching" in name:
            mira2 = np.load(path, allow_pickle=True).item()
            source = path
        if "pgrid" in name:
            pgrid = np.load(path, allow_pickle=True) / 1e2
        if "apriori" in name:
            apriori = np.load(path, allow_pickle=True)

    for k, v in mira2.items():
        dt.append(k)
        date.append(k.date())
        time.append(k.time())

        p = v["apriori"] * v["x"] * 1e6
        prod.append(p)

        e = v["ss"] + v["eo"]
        err.append(e)

    mira2_data = MIRA2Data(
        source=source,
        dt=np.array(dt),
        prod=np.array(prod),
        err=np.array(err),
        pgrid=pgrid,
        apriori=apriori,
        date=np.array(date),
        time=np.array(time),
    )
    dt = []
    date = []
    time = []
    prod = []
    lons = []
    lats = []

    for path in mls_paths:
        name = path.name
        if "smooth" in name:
            mls = np.load(path, allow_pickle=True).item()
            source = path
        if "pgrid" in name:
            pgrid = np.load(path, allow_pickle=True) / 1e2

    for k, v in mls.items():
        dt.append(k)
        date.append(k.date())
        time.append(k.time())

        p = v["O3smooth"] * 1e6
        lon = v["lon"]
        lat = v["lat"]
        prod.append(p)
        lons.append(lon)
        lats.append(lat)

    mls_data = MLSData(
        source=source,
        dt=np.array(dt),
        prod=np.array(prod),
        lon=np.array(lons),
        lat=np.array(lats),
        date=np.array(date),
        time=np.array(time),
        pgrid=pgrid,
    )

    return mira2_data, mls_data


def read_edgedata(which: str, severity: int) -> EdgeData:
    root = Path("/home/ric/Data")
    all_files = get_egdefiles(root=root)
    target = None
    for file in all_files:
        if which in file.name:
            target = file
    assert target is not None

    edgedata = parse_edgefile(file=target)
    filt_edgedata = filter_edgedata(data=edgedata, severity=severity)
    return filt_edgedata


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


m2day, mlsday = read_profiles(which="day")
m2night, mlsnight = read_profiles(which="night")

print(m2day.dt)
print(m2night.dt)
edge475 = read_edgedata(which="475", severity=2)
i4, i7, i17 = get_pressure_level_index(m2day.pgrid)

# day data
daydata_4 = PressureLevelData(
    pl=m2day.pgrid[i4],
    mira2_dt=m2day.dt,
    mls_dt=mlsday.dt,
    mira2=np.array([prod[i4] for prod in m2day.prod]),
    mls=np.array([prod[i4] for prod in mlsday.prod]),
    err=np.array([err[i4] for err in m2day.err]),
    mira2_apriori=np.full_like(m2day.dt, m2day.apriori[i4] * 1e6),
    mls_lat=mlsday.lat,
    mls_lon=mlsday.lon,
)
daydata_7 = PressureLevelData(
    pl=m2day.pgrid[i7],
    mira2_dt=m2day.dt,
    mls_dt=mlsday.dt,
    mira2=np.array([prod[i7] for prod in m2day.prod]),
    mls=np.array([prod[i7] for prod in mlsday.prod]),
    err=np.array([err[i7] for err in m2day.err]),
    mira2_apriori=np.full_like(m2day.dt, m2day.apriori[i7] * 1e6),
    mls_lat=mlsday.lat,
    mls_lon=mlsday.lon,
)
daydata_17 = PressureLevelData(
    pl=m2day.pgrid[i17],
    mira2_dt=m2day.dt,
    mls_dt=mlsday.dt,
    mira2=np.array([prod[i17] for prod in m2day.prod]),
    mls=np.array([prod[i17] for prod in mlsday.prod]),
    err=np.array([err[i17] for err in m2day.err]),
    mira2_apriori=np.full_like(m2day.dt, m2day.apriori[i17] * 1e6),
    mls_lat=mlsday.lat,
    mls_lon=mlsday.lon,
)

# night data
nightdata_4 = PressureLevelData(
    pl=m2night.pgrid[i4],
    mira2_dt=m2night.dt,
    mls_dt=mlsnight.dt,
    mira2=np.array([prod[i4] for prod in m2night.prod]),
    mls=np.array([prod[i4] for prod in mlsnight.prod]),
    err=np.array([err[i4] for err in m2night.err]),
    mira2_apriori=np.full_like(m2day.dt, m2day.apriori[i4] * 1e6),
    mls_lat=mlsnight.lat,
    mls_lon=mlsnight.lon,
)
nightdata_7 = PressureLevelData(
    pl=m2night.pgrid[i7],
    mira2_dt=m2night.dt,
    mls_dt=mlsnight.dt,
    mira2=np.array([prod[i7] for prod in m2night.prod]),
    mls=np.array([prod[i7] for prod in mlsnight.prod]),
    err=np.array([err[i7] for err in m2night.err]),
    mira2_apriori=np.full_like(m2day.dt, m2day.apriori[i7] * 1e6),
    mls_lat=mlsnight.lat,
    mls_lon=mlsnight.lon,
)
nightdata_17 = PressureLevelData(
    pl=m2night.pgrid[i17],
    mira2_dt=m2night.dt,
    mls_dt=mlsnight.dt,
    mira2=np.array([prod[i17] for prod in m2night.prod]),
    mls=np.array([prod[i17] for prod in mlsnight.prod]),
    err=np.array([err[i17] for err in m2night.err]),
    mira2_apriori=np.full_like(m2day.dt, m2day.apriori[i17] * 1e6),
    mls_lat=mlsnight.lat,
    mls_lon=mlsnight.lon,
)


def plot_levels(which: str, edgefile: EdgeData):
    if which == "day":
        data_4 = daydata_4
        data_7 = daydata_7
        data_17 = daydata_17

    else:
        data_4 = nightdata_4
        data_7 = nightdata_7
        data_17 = nightdata_17

    fig = go.Figure()
    fig.update_layout(template="simple_white", width=1600, height=800)

    fig.add_trace(
        go.Scatter(
            x=edgefile.date,
            y=np.full_like(edgefile.date, 9.8),
            visible="legendonly",
            mode="markers",
            marker=dict(color="white", line=dict(width=2, color="DarkSlateGrey")),
            legendgroup="edge475",
            name="MIRA2 inside polarvortex (475)",
        )
    )

    # day 4 hPa
    fig.add_trace(
        go.Scatter(
            x=data_4.mira2_dt,
            y=data_4.mira2,
            visible="legendonly",
            mode="markers",
            error_y=dict(type="data", array=data_4.err, visible=True),
            name="MIRA2 4 hPa",
            legendgroup="4 hPa",
            line=dict(color="black"),
            marker=dict(color="black"),
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=data_4.mira2_dt,
            y=data_4.mira2_apriori,
            visible="legendonly",
            mode="lines",
            name="MIRA2 4 hPa apriori",
            legendgroup="4 hPa",
            line=dict(color="green"),
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=data_4.mls_dt.flatten(),
            y=data_4.mls.flatten(),
            visible="legendonly",
            name="MLS 4 hPa",
            legendgroup="4 hPa",
            text=[
                f"lon:{lon}, lat:{lat}"
                for lon, lat in zip(data_4.mls_lon, data_4.mls_lat)
            ],
            line=dict(color="red"),
        ),
    )

    # day 7 hPa
    fig.add_trace(
        go.Scatter(
            x=data_7.mira2_dt,
            y=data_7.mira2,
            visible="legendonly",
            mode="markers",
            error_y=dict(type="data", array=data_7.err, visible=True),
            name="MIRA2 7 hPa",
            legendgroup="7 hPa",
            line=dict(color="black"),
            marker=dict(color="black"),
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=data_7.mira2_dt,
            y=data_7.mira2_apriori,
            visible="legendonly",
            mode="lines",
            name="MIRA2 7 hPa apriori",
            legendgroup="7 hPa",
            line=dict(color="green"),
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=data_7.mls_dt.flatten(),
            y=data_7.mls.flatten(),
            visible="legendonly",
            name="MLS 7 hPa",
            legendgroup="7 hPa",
            text=[
                f"lon:{lon}, lat:{lat}"
                for lon, lat in zip(data_7.mls_lon, data_7.mls_lat)
            ],
            line=dict(color="red"),
        ),
    )
    # day 17 hPa
    fig.add_trace(
        go.Scatter(
            x=data_17.mira2_dt,
            y=data_17.mira2,
            visible="legendonly",
            mode="markers",
            error_y=dict(type="data", array=data_17.err, visible=True),
            name="MIRA2 17 hPa",
            legendgroup="17 hPa",
            line=dict(color="black"),
            marker=dict(color="black"),
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=data_17.mira2_dt,
            y=data_17.mira2_apriori,
            visible="legendonly",
            mode="lines",
            name="MIRA2 17 hPa apriori",
            legendgroup="17 hPa",
            line=dict(color="green"),
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=data_17.mls_dt.flatten(),
            y=data_17.mls.flatten(),
            visible="legendonly",
            name="MLS 17 hPa",
            legendgroup="17 hPa",
            text=[
                f"lon:{lon}, lat:{lat}"
                for lon, lat in zip(data_17.mls_lon, data_17.mls_lat)
            ],
            line=dict(color="red"),
        ),
    )

    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        title_text="Date",
        range=[
            datetime_date(year=2019, month=9, day=27),
            datetime_date(year=2020, month=5, day=5),
        ],
        tickfont=dict(size=16),
        title_font=dict(size=18),
    )
    fig.update_yaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        range=[0, 11],
        title_text="VMR [ppm]",
        tickfont=dict(size=16),
        title_font=dict(size=18),
    )
    fig.update_traces(showlegend=True)
    fig.write_html(f"{which}.html")


plot_levels(which="day", edgefile=edge475)
plot_levels(which="night", edgefile=edge475)
