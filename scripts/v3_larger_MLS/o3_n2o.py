from h5py import File
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta, date
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


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


def o3_files():
    home = Path.home()
    mlsdir = home / "MLS"
    ddgdir = mlsdir / "O3"
    gen = ddgdir.glob(pattern="*.he5")
    files = []
    for file in gen:
        name = file.name
        if "c01" in name:
            files.append(file)

    return np.array(files)


def get_species_files(species):
    home = Path.home()
    mlsdir = home / "MLS"
    ddgdir = mlsdir / species
    gen = ddgdir.glob(pattern="*.he5")
    files = []
    for file in gen:
        name = file.name
        if "c01" in name:
            files.append(file)

    return np.array(files)


def n2o_files():
    home = Path.home()
    mlsdir = home / "MLS"
    ddgdir = mlsdir / "N2O"
    gen = ddgdir.glob(pattern="*.he5")
    files = []
    for file in gen:
        name = file.name
        if "c01" in name:
            files.append(file)

    return np.array(files)


def read_dgg_files(files):
    dts = []
    for path in files:
        with File(path, "r") as fp:
            hdfeos = fp["HDFEOS"]
            swaths = hdfeos["SWATHS"]
            potvort = swaths["PotentialVorticity"]
            eql = swaths["PVEquivalentLatitude"]["Data Fields"]["PVEquivalentLatitude"][
                ()
            ]
            data = potvort["Data Fields"]["PotentialVorticity"][()]
            pvs = data[:, 16]
            geos = potvort["Geolocation Fields"]
            pressure = geos["Pressure"][()]
            print(pressure[12])
            latitudes = geos["Latitude"][()]
            longitudes = geos["Longitude"][()]
            timestamps = geos["Time"][()]
            start = date(2019, 11, 20)
            stop = date(2019, 11, 25)

            for pv, lat, lon, ts in zip(pvs, latitudes, longitudes, timestamps):
                spv = pv * 1e5
                try:
                    dt = make_datetime(ts)
                except OverflowError:
                    continue
                d = dt.date()
                if (
                    lat >= 70
                    and spv >= 3.6
                    and start <= d <= stop
                    and -180 <= lon <= 180
                ):
                    dts.append(dt)

    return dts


def get_species_data(files, species):
    dct = {}
    for path in files:
        with File(path, "r") as fp:
            hdfeos = fp["HDFEOS"]
            swaths = hdfeos["SWATHS"]
            temp = swaths[species]
            data = temp["Data Fields"][species][()]
            geos = temp["Geolocation Fields"]
            latitudes = geos["Latitude"][()]
            longitudes = geos["Longitude"][()]
            timestamps = geos["Time"][()]
            pressure = geos["Pressure"][()]

            for prod, ts, lat, lon in zip(data, timestamps, latitudes, longitudes):
                try:
                    dt = make_datetime(ts)
                except OverflowError:
                    continue

                if lat >= 70:
                    dct[dt] = [prod, pressure]

    return dct


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


# dmp = "/home/ric/Downloads/MLS-Aura_L2EDMP-GEOS5294-v201_v05-00-c01_2019d001.he5"
dmpfiles = dmp_files()
n2ofiles = n2o_files()
dts = read_dgg_files(dmpfiles)
n2o = get_species_data(get_species_files(species="N2O"), species="N2O")
o3 = get_species_data(get_species_files(species="O3"), species="O3")

o3lev = []
n2olev = []
for dt in dts:
    try:
        prod_n2o = n2o[dt]
        prod_o3 = o3[dt]
        o3i = 16
        n2oi = 8
        o3lev.append(prod_o3[0][o3i])
        n2olev.append(prod_n2o[0][n2oi])
    except KeyError:
        continue


# o3 and n2o are your arrays (same length)
o3 = np.array(o3lev) * 1e6
n2o = np.array(n2olev) * 1e9

fig = plt.figure(figsize=(10, 10))
gs = GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.scatter(n2o, o3)
fig.savefig(f"O3-N2O_{prod_o3[1][o3i]}.pdf")
