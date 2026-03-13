from ozone.io import get_downloadsdir, get_egdefiles, get_datadir, get_home_data
from ozone.analysis import parse_edgefile, filter_edgedata, binning, fit_n2o_o3
from scipy.interpolate import interp1d
from pathlib import Path
from h5py import File
from dataclasses import dataclass
from datetime import date, datetime, timedelta


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


@dataclass
class DMPdata:
    pressure: np.ndarray
    timestamp: np.ndarray
    theta: np.ndarray
    altitude: np.ndarray


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


def calc_pottemp(t, p):
    kap = 0.286

    pot = t * (1e5 / p) ** kap
    return pot


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
        timestamps = []

        altitude = swaths["Altitude/Data Fields/Altitude"][()]
        pveql = swaths["PVEquivalentLatitude"]
        theta = swaths["Theta/Data Fields/Theta"][()]
        pressure = pveql["Geolocation Fields/Pressure"][()]
        latitude = pveql["Geolocation Fields/Latitude"][()]
        for ts in pveql["Geolocation Fields/Time"][()]:
            try:
                timestamp = make_datetime(ts)
                timestamps.append(timestamp)
            except OverflowError:
                timestamps.append(np.nan)

        timestamps = np.array(timestamps)
        mask = latitude >= 60

        return DMPdata(
            pressure=pressure,
            theta=theta[mask],
            altitude=altitude[mask],
            timestamp=timestamps[mask],
        )


def get_dmpdata(files):
    res = {}
    ptargetp = get_datadir() / "m2pres.npz"
    ptarget = np.load(ptargetp)["pressure"]
    logtgt = np.log(ptarget)
    for file in files:
        data = read_dmp(file)
        altitudes = data.altitude
        thetas = data.theta
        timestamps = data.timestamp
        pressure = data.pressure

        for ts, alts, theta in zip(timestamps, altitudes, thetas):
            try:
                logsrc = np.log(pressure)
                interp_theta = interp1d(
                    logsrc,
                    theta,
                    bounds_error=False,
                    fill_value=np.nan,
                )
                ntheta = interp_theta(logtgt)
                res[ts] = {
                    "date": ts.date(),
                    "theta_interp": ntheta,
                    "pressure_interp": pressure,
                }
            except AttributeError:
                continue

    return res


def get_dates_in_vortex(edgefiles, severity):
    f435 = edgefiles[4]
    f475 = edgefiles[1]
    f550 = edgefiles[0]
    edge435 = filter_edgedata(parse_edgefile(f435), severity=severity).date
    edge475 = filter_edgedata(parse_edgefile(f475), severity=severity).date
    edge550 = filter_edgedata(parse_edgefile(f550), severity=severity).date

    common_dates = sorted(list(set(edge435) & set(edge475) & set(edge550)))
    return np.array(common_dates)


m2p = get_home_data() / "MIRA2_v3_screened_matching.npy"
n2op = get_home_data() / "MLS_N2O_screened_matching.npy"
m2prespath = get_datadir() / "m2pres.npz"
edgefiles = get_egdefiles("/home/ric/Data/edge/")

mlsN2O = np.load(n2op, allow_pickle=True).item()
m2O3 = np.load(m2p, allow_pickle=True).item()
ptarget = np.load(m2prespath)["pressure"]
logtgt = np.log(ptarget)
inside_dates = get_dates_in_vortex(edgefiles, severity=1)

dmpfiles = dmp_files()
dmpdata = get_dmpdata(dmpfiles)
for k, v in m2O3.items():
    logsrc = np.log(v["pmeas"])
    ptarget = v["pgrid"]
    logtgt = np.log(ptarget)
    xsrc = v["tmeas"]

    interp = interp1d(
        logsrc,
        xsrc,
        bounds_error=False,
        fill_value=np.nan,
    )

    ttar = interp(logtgt)
    pot = calc_pottemp(ttar, ptarget)
    m2O3[k]["theta"] = pot

for dt, val in dmpdata.items():
    try:
        mlsN2O[dt]["theta_interp"] = val["theta_interp"]
        mlsN2O[dt]["date"] = val["date"]
        ptarget = np.load(get_datadir() / "m2pres.npz")["pressure"]
        psource = mlsN2O[dt]["pressure"]

        logsrc = np.log(psource)
        logtgt = np.log(ptarget)
        vmr = mlsN2O[dt]["N2O"]
        prc = mlsN2O[dt]["precision"]
        interp_vmr = interp1d(logsrc, vmr, bounds_error=False, fill_value=np.nan)
        interp_prc = interp1d(logsrc, prc, bounds_error=False, fill_value=np.nan)

        new_vmr = interp_vmr(logtgt)
        new_prc = interp_prc(logtgt)

        mlsN2O[dt]["n2o_interp"] = new_vmr
        mlsN2O[dt]["precision_interp"] = new_prc
        mlsN2O[dt]["pressure_interp"] = ptarget

    except KeyError:
        continue

mlsN2Oinside = {}
start = date(2019, 11, 25)
stop = date(2019, 12, 25)

n2ovalues = []
n2oerrs = []
o3values = []
o3errs = []

m2dates = np.array([dt.date() for dt in m2O3])
m2dts = np.array([dt for dt in m2O3])

for dt, val in mlsN2O.items():
    d = dt.date()
    if d in inside_dates and start <= d <= stop:
        theta = val["theta_interp"]
        vmr = val["n2o_interp"]
        n2oerr = val["precision_interp"]
        pressure = val["pressure_interp"]
        mask = (theta >= 435) & (theta <= 550)
        indecies = np.where(m2dates == d)[0]

        for dt in m2dts[indecies]:
            m2theta = m2O3[dt]["theta"]
            m2mask = (m2theta >= 435) & (m2theta <= 550)
            cov = m2O3[dt]["S_phys"]
            dcov = np.diag(cov)
            o3values.extend(m2O3[dt]["x_phys"][mask].flatten() / 1e6)
            o3errs.extend(np.sqrt(dcov)[mask].flatten() / 1e6)
            n2ovalues.extend(vmr[mask].flatten())
            n2oerrs.extend(n2oerr[mask].flatten())


xv = np.array([d for d in n2ovalues])
yv = np.array([d for d in o3values])
xe = np.array([d for d in n2oerrs])
ye = np.array([d for d in o3errs])
xval, yval, xerr, yerr = binning(
    xdata=xv,
    ydata=yv,
    xerr=xe,
    yerr=ye,
    n_bins=20,
    min_bin_points=2,
)
res, fit = fit_n2o_o3(
    x=xv,
    y=yv,
    errx=xe,
    erry=ye,
    filename="fit_MIRA2.npz",
    xmin=5e-9,
    xmax=185e-9,
)
xfit = fit.xfit
yfit = fit.yfit
sfit = fit.sfit

fig = plt.figure(figsize=(13, 8))
gs = GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.scatter(n2ovalues, o3values)
ax.scatter(xval, yval)
ax.plot(xfit, yfit)
fig.savefig("test_mira2_tracer.pdf")

m2path = get_home_data() / "MIRA2_v3_screened_matching_w_theta.npy"
mlsN2Opath = get_home_data() / "MLS_N2O_screened_matching_interp.npy"
np.save(m2path, m2O3, allow_pickle=True)
np.save(mlsN2Opath, mlsN2O, allow_pickle=True)
