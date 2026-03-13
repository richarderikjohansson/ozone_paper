from h5py import File
import numpy as np

from datetime import date
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, date, timedelta
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from ozone.io import get_downloadsdir, get_datadir, get_home_data


@dataclass
class DMPdata:
    eql: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    pressure: np.ndarray
    timestamp: np.ndarray
    theta: np.ndarray
    pv: np.ndarray
    spv: np.ndarray
    gradpv: np.ndarray
    altitude: np.ndarray


@dataclass
class InsideVortexData:
    date: date
    timestamp: datetime
    pressure: np.ndarray
    theta: np.ndarray
    eqlmask: np.ndarray


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


def read_dmp(fp):
    with File(fp, "r") as fh:
        data = fh["HDFEOS"]
        swaths = data["SWATHS"]
        timestamps = []

        pv = swaths["PotentialVorticity/Data Fields/PotentialVorticity"][()]
        spv = swaths["ScaledPV/Data Fields/ScaledPV"][()]
        altitude = swaths["Altitude/Data Fields/Altitude"][()]
        gradpv = swaths["HorizontalPVGradient/Data Fields/HorizontalPVGradient"][()]
        pveql = swaths["PVEquivalentLatitude"]
        eql = pveql["Data Fields/PVEquivalentLatitude"][()]
        theta = swaths["Theta/Data Fields/Theta"][()]
        latitude = pveql["Geolocation Fields/Latitude"][()]
        longitude = pveql["Geolocation Fields/Longitude"][()]
        pressure = pveql["Geolocation Fields/Pressure"][()]
        for ts in pveql["Geolocation Fields/Time"][()]:
            try:
                timestamp = make_datetime(ts)
                timestamps.append(timestamp)
            except OverflowError:
                timestamps.append(np.nan)

        timestamps = np.array(timestamps)

        return DMPdata(
            eql=eql,
            latitude=latitude,
            longitude=longitude,
            pressure=pressure,
            timestamp=timestamps,
            theta=theta,
            pv=pv,
            gradpv=gradpv,
            spv=spv,
            altitude=altitude,
        )


def get_dates(files):
    res = {}
    for file in files:
        data = read_dmp(file)
        mask = data.latitude >= 40
        eqls = data.eql[mask]
        timestamps = data.timestamp[mask]
        thetas = data.theta[mask]
        spvs = data.spv[mask] * 1e6

        for eql, timestamp, theta, spv in zip(eqls, timestamps, thetas, spvs):
            # edgemask = (eql >= 70) & (eql <= 80) & (theta >= 400) & (theta <= 600)
            edgemask = (spv >= 150) & (spv <= 1000) & (theta >= 460) & (theta <= 580)
            if not edgemask.sum() == 0:
                d = timestamp.date()
                pressure = data.pressure
                altitude = data.altitude
                res[timestamp] = InsideVortexData(
                    date=d,
                    timestamp=timestamp,
                    theta=theta,
                    eqlmask=edgemask,
                    pressure=pressure,
                )
                res[timestamp] = {
                    "date": d,
                    "timestamp": timestamp,
                    "theta": theta,
                    "edgemask": edgemask,
                    "pressure": pressure,
                    "altitude": altitude,
                }

    return res


def get_dmpdata(latlim, files, thetalims=None, spvlims=None):
    res = {}
    ptargetp = get_datadir() / "m2pres.npz"
    ptarget = np.load(ptargetp)["pressure"]
    logtgt = np.log(ptarget)
    for file in files:
        data = read_dmp(file)
        latmask = data.latitude >= latlim
        latitudes = data.latitude[latmask]
        longitudes = data.longitude[latmask]
        eqls = data.eql[latmask]
        timestamps = data.timestamp[latmask]
        altitudes = data.altitude[latmask]
        pressures = data.pressure
        logsrc = np.log(pressures)
        thetas = data.theta[latmask]
        gradpv = data.gradpv[latmask]
        pvs = data.pv[latmask]
        spvs = data.spv[latmask] * 1e6

        if thetalims is not None and spvlims is not None:
            for i, (ts, theta, eql) in enumerate(zip(timestamps, thetas, eqls)):
                thetamask = (thetalims[0] <= theta) & (theta <= thetalims[1])
                c = thetamask.sum()
                # interp_theta = interp1d(logsrc, theta)
                # theta_new = interp_theta(logtgt)

                # interp_eql = interp1d(theta, eql)
                # eql_new = interp_eql(theta_new)
                edgemask = (
                    (eql >= 72)
                    & (eql < 90)
                    & (theta >= thetalims[0])
                    & (theta <= thetalims[1])
                )
                if edgemask.sum() >= 1:
                    try:
                        d = ts.date()
                        res[ts] = {
                            "date": d,
                            "latitude": latitudes[i],
                            "longitude": longitudes[i],
                            "eql": eql,
                            #             "eql_interp": eql_new,
                            "pressure": pressures,
                            "pressure_interp": ptarget,
                            "theta": theta,
                            #             "theta_interp": theta_new,
                            "edgemask": edgemask,
                        }
                    except AttributeError:
                        continue

    return res


dmpfiles = dmp_files()
res = get_dmpdata(latlim=55, files=dmpfiles, thetalims=(400, 600), spvlims=(150, 2000))
np.save(get_home_data() / "dmpdata_kir.npy", res, allow_pickle=True)

inside = get_dates(dmpfiles)
dates = [val["date"] for val in inside.values()]
pressure = np.array([val["pressure"] for val in inside.values()])

insidepath = get_downloadsdir() / "inside.npy"
datespath = get_downloadsdir() / "dates.npz"

np.save(insidepath, inside, allow_pickle=True)
np.save(datespath, dates)
