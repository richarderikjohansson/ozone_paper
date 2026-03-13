import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime, timedelta, date
from dataclasses import dataclass
from pathlib import Path
from h5py import File
from types import SimpleNamespace

from ozone.analysis import fit_n2o_o3, poly4_odr, match_tracers, binning
from ozone.io import get_downloadsdir, get_home_data
from ozone._const import COLORS


@dataclass
class DMPdata:
    eql: np.ndarray
    pressure: np.ndarray
    timestamp: np.ndarray
    measdate: date
    latitude: float


def dmp_files():
    home = Path.home()
    mlsdir = home / "MLS"
    ddgdir = mlsdir / "DMP"
    gen = ddgdir.glob(pattern="*.he5")
    files = []
    for file in gen:
        name = file.name
        if "c01" in name:
            files.append(file)

    return np.array(files)


def make_datetime(ts: float) -> datetime:
    """Function to create datetime objects

    This function takes the data in the 'TIME' field from
    the .he5 MLS files and converts the time to a datetime
    object and returns a numpy ndarray with these datetime
    objects

    Args:
        seconds_array: array associated with the 'TIME' field

    Returns:
        numpy array with datetime objects
    """
    epoch = datetime(1993, 1, 1, 0, 0, 0)

    dt = epoch + timedelta(seconds=ts)
    dt = dt.replace(microsecond=0)
    return dt


def read_dmp(fp):
    with File(fp, "r") as fh:
        data = fh["HDFEOS"]
        swaths = data["SWATHS"]

        eqlset = swaths["PVEquivalentLatitude"]
        eql = eqlset["Data Fields/PVEquivalentLatitude"][()]
        pressure = eqlset["Geolocation Fields/Pressure"][()]
        latitude = eqlset["Geolocation Fields/Latitude"][()]
        timestamps = eqlset["Geolocation Fields/Time"][()]
        timestamp = np.array([make_datetime(ts) for ts in timestamps])
        measdate = timestamp[0].date()

        return DMPdata(
            eql=eql,
            pressure=pressure,
            timestamp=timestamp,
            measdate=measdate,
            latitude=latitude,
        )


def filter_measurements(tracedata, start, stop):
    tracedts = tracedata.keys()
    dct = {}
    for dt in tracedts:
        measdate = dt.date()
        if start <= measdate <= stop:
            dct[dt] = tracedata[dt]

    return dct


xdata_file = get_downloadsdir() / "N2O_tracer_screened.npy"
xdata = np.load(xdata_file, allow_pickle=True).item()

ydata_file = get_downloadsdir() / "O3_tracer_screened.npy"
ydata = np.load(ydata_file, allow_pickle=True).item()


start = date(2019, 12, 1)
stop = date(2019, 12, 15)
tracedata = match_tracers(xdata=xdata, ydata=ydata)
filtered = filter_measurements(tracedata, start, stop)
print(len(filtered))
xval = []
yval = []
xerr = []
yerr = []

for val in filtered.values():
    xval.extend(val.xval)
    yval.extend(val.yval)
    xerr.extend(val.xerr)
    yerr.extend(val.yerr)
temp = SimpleNamespace(
    xval=np.array(xval),
    yval=np.array(yval),
    xerr=np.array(xerr),
    yerr=np.array(yerr),
)
tracedata = temp
dts = [dt for dt in filtered.keys()]


xval_binned, yval_binned, xerr_binned, yerr_binned = binning(
    xdata=tracedata.xval,
    ydata=tracedata.yval,
    xerr=tracedata.xerr,
    yerr=tracedata.yerr,
    n_bins=200,
    core_percentile=(50, 50),
    min_bin_points=15,
)

res, fit = fit_n2o_o3(
    x=xval_binned,
    y=yval_binned,
    errx=xerr_binned,
    erry=yerr_binned,
    filename="fit_filtered.npz",
)
xfit = fit.xfit
yfit = fit.yfit
sfit = fit.sfit

fig = plt.figure(figsize=(13, 10))
gs = GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.errorbar(
    x=tracedata.xval * 1e9,
    y=tracedata.yval * 1e6,
    xerr=tracedata.xerr * 1e9,
    yerr=tracedata.yerr * 1e6,
    fmt="o",
    color=COLORS.overlay,
    alpha=0.2,
    label=r"$O_3/N_2O$",
    zorder=1,
)
ax.errorbar(
    x=xval_binned * 1e9,
    y=yval_binned * 1e6,
    xerr=xerr_binned * 1e9,
    yerr=yerr_binned * 1e6,
    fmt="o",
    color=COLORS.red,
    alpha=0.2,
    label=r"$O_3/N_2O$ (binned)",
    zorder=2,
)
ax.plot(
    xfit * 1e9,
    yfit * 1e6,
    color=COLORS.mantle,
    linewidth=3,
    label=r"$Fit$ $\pm \sigma$",
    zorder=4,
)
ax.fill_between(
    xfit * 1e9,
    y1=(yfit + sfit) * 1e6,
    y2=(yfit - sfit) * 1e6,
    color=COLORS.mantle,
    alpha=0.25,
    zorder=3,
)
ax.minorticks_on()
ax.grid(which="both", alpha=0.1)
ax.set_ylabel(r"$O_3$ / ppmv", fontsize=14)
ax.set_xlabel(r"$N_2O$ / ppbv", fontsize=14)
ax.tick_params(labelsize=13)
ax.legend()
ax.set_title(rf"{dts[0]} - {dts[-1]} ($\theta$: 400 - 600 K and $\phi_e \geq 80$) ")
fig.savefig("scatter_and_fit_filtered_400-600_80.pdf")
