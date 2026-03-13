from pathlib import Path
import h5py
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
from .io import get_exportdir, get_datadir, get_downloadsdir
from .logger import get_logger
from .utils import fill_nans
from haversine import haversine, Unit
from .screening import MLSScreener
import logging
import yaml


def make_datetime(seconds_array: np.ndarray) -> datetime:
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
    dtarr = []

    for sec in seconds_array:
        dt = epoch + timedelta(seconds=sec)
        dt = dt.replace(microsecond=0)
        dtarr.append(dt)
    return np.array(dtarr)


class MLSFindAndMake:
    """
    Find MLS files and make .npy file from these
    """

    def __init__(self, root: str, logger):
        """Init constructor

        Args:
            root: Root directory for the MLS files
            make: Boolean is a file will be made
            radii: Geographical radius from Kiruna
        """
        self.root = Path(root).resolve()
        self.name = self.root.name
        if self.name == "T":
            self.name = "Temperature"
        self.loc = (67.84, 20.41)
        self.radii = 400
        self.logger = logger
        self.find_mls()
        self.make_mls()

    def find_mls(self):
        """Method to find the files

        Recursively searches through the root directory for MLS files
        """
        self.logger.info(f"Finding MLS files from {self.root}")
        files = self.root.rglob(pattern="*.he5")
        self.files = sorted([file for file in files])

    def make_mls(self):
        """Method to make the .npy file

        This method extracts all necessary datasets to do a proper
        analysis using MLS data. It also employs the haversine function
        to limit the data so only data within a certain radii from Kiruna
        will be extracted
        """
        files = []
        umlsdct = {}
        edir = get_exportdir()
        ddir = get_datadir()
        daterange = np.load(ddir / "daterange.npy", allow_pickle=True)
        start = daterange[0]
        end = daterange[-1]

        for file in tqdm(self.files, desc=f"Getting MLS {self.name} data"):
            with h5py.File(file, "r") as fh:
                data = fh["HDFEOS"]
                swaths = data["SWATHS"]
                prod = swaths[self.name]
                datafields = prod["Data Fields"]
                geolocfields = prod["Geolocation Fields"]

                self.get_data(datafields)
                self.get_geoloc(geolocfields)

                for i, _ in enumerate(self.dt):
                    # make sure that only real coords is present
                    if (
                        #                        self.lat[i] >= 65.5
                        self.lat[i] <= 90 and self.lat[i] >= -90
                        #                        and self.lon[i] >= 18.5
                        #                        and self.lon[i] <= 25
                    ):
                        mls = (self.lat[i], self.lon[i])
                        haversinedist = haversine(
                            self.loc,
                            mls,
                            unit=Unit.KILOMETERS,
                        )
                    else:
                        continue

                    if (
                        haversinedist <= self.radii
                        and start <= self.dt[i].date() <= end
                    ):
                        umlsdct[self.dt[i]] = {
                            f"{self.name}": self.prod[i],
                            "convergence": self.convergence[i],
                            "l2precision": self.l2_precision[i],
                            "l2value": self.l2_value[i],
                            "precision": self.precision[i],
                            "quality": self.quality[i],
                            "status": self.status[i],
                            "lat": self.lat[i],
                            "lon": self.lon[i],
                            "pressure": self.p_grid,
                            "time": self.time[i],
                        }
                        files.append(file)

        sorted_keys = sorted(umlsdct.keys())
        mlsdct = {key: umlsdct[key] for key in sorted_keys}
        sdict = fill_nans(mlsdct)
        mdict = {
            "product": self.name,
            "make_date": datetime.now(),
            "haversine": self.radii,
            "sources": files,
        }

        savepath = edir / f"{self.name}.npy"
        metapath = edir / f"{self.name}.meta.npy"

        np.save(savepath.resolve(), sdict, allow_pickle=True)
        np.save(metapath.resolve(), mdict, allow_pickle=True)
        self.logger.info(f"Saved data into {savepath}")

    def get_data(self, datafields):
        """Method to get all data from data fields

        This is a helper method used to extract all the necessary
        data from 'Data Fields' in the MLS file

        Args:
            datafields: dataset with data fields
        """
        self.prod = datafields[self.name][()]
        self.convergence = datafields["Convergence"][()]
        self.l2_precision = datafields["L2gpPrecision"][()]
        self.l2_value = datafields["L2gpValue"][()]
        self.precision = datafields[f"{self.name}Precision"][()]
        self.quality = datafields["Quality"][()]
        self.status = datafields["Status"][()]

    def get_geoloc(self, geolocfields):
        """Method to get all data from geo-location fields

        Helper method to extract the necessary data from the
        'Geolocation Fields' in the MLS file

        Args:
            geolocfields: dataset with geolocation fields
        """
        self.lat = geolocfields["Latitude"][()]
        self.lon = geolocfields["Longitude"][()]
        self.p_grid = geolocfields["Pressure"][()]
        self.time = geolocfields["Time"][()]
        self.dt = make_datetime(self.time)


class MLSFindAndMakeTracer:
    def __init__(self, root, logger, latbound=(90, 50), lonbound=(-180, 180)):
        self.root = Path(root).resolve()
        self.tracers = ["O3", "N2O", "ClO", "T"]
        self.latmax = latbound[0]
        self.latmin = latbound[1]
        self.lonmax = lonbound[1]
        self.lonmin = lonbound[0]
        self.logger = logger
        self.make_mls()

    def find_mls(self, tracer):
        """Method to find the files

        Recursively searches through the root directory for MLS files
        """
        self.logger.info(f"Finding MLS files from {self.root}")
        tracerdir = self.root / tracer
        files = tracerdir.rglob(pattern="*.he5")
        self.files = sorted([file for file in files])

    def make_mls(self):
        """Method to make the .npy file

        This method extracts all necessary datasets to do a proper
        analysis using MLS data. It also employs the haversine function
        to limit the data so only data within a certain radii from Kiruna
        will be extracted
        """
        for tracer in self.tracers:
            self.find_mls(tracer=tracer)
            if tracer == "T":
                self.name = "Temperature"
            else:
                self.name = tracer
            files = []
            umlsdct = {}
            edir = get_downloadsdir()
            ddir = get_datadir()
            daterange = np.load(ddir / "daterange.npy", allow_pickle=True)
            start = daterange[0]
            end = daterange[-1]

            for file in tqdm(self.files, desc=f"Getting MLS {self.name} data"):
                with h5py.File(file, "r") as fh:
                    data = fh["HDFEOS"]
                    swaths = data["SWATHS"]
                    prod = swaths[self.name]
                    datafields = prod["Data Fields"]
                    geolocfields = prod["Geolocation Fields"]

                    self.get_data(datafields)
                    self.get_geoloc(geolocfields)

                    for i, _ in enumerate(self.dt):
                        # make sure that only real coords is present
                        if (
                            self.lat[i] >= self.latmin
                            and self.lat[i] <= self.latmax
                            and self.lon[i] >= self.lonmin
                            and self.lon[i] <= self.lonmax
                            and start <= self.dt[i].date() <= end
                        ):
                            umlsdct[self.dt[i]] = {
                                f"{self.name}": self.prod[i],
                                "convergence": self.convergence[i],
                                "l2precision": self.l2_precision[i],
                                "l2value": self.l2_value[i],
                                "precision": self.precision[i],
                                "quality": self.quality[i],
                                "status": self.status[i],
                                "lat": self.lat[i],
                                "lon": self.lon[i],
                                "pressure": self.p_grid,
                                "time": self.time[i],
                            }
                            files.append(file)
                        else:
                            continue

            sorted_keys = sorted(umlsdct.keys())
            mlsdct = {key: umlsdct[key] for key in sorted_keys}
            sdict = fill_nans(mlsdct)
            mdict = {
                "product": self.name,
                "make_date": datetime.now(),
                "sources": files,
            }

            savepath = edir / f"{self.name}_tracer_screened"
            screenfile = get_datadir() / f"{self.name}.yaml"
            with open(screenfile, "r") as fh:
                screen = yaml.load(fh, Loader=yaml.SafeLoader)

            screener = MLSScreener(
                data=sdict, meta=mdict, screen=screen, logger=self.logger, winter=True
            )
            screener.save_screened_data(filename=savepath)

    def get_data(self, datafields):
        """Method to get all data from data fields

        This is a helper method used to extract all the necessary
        data from 'Data Fields' in the MLS file

        Args:
            datafields: dataset with data fields
        """
        self.prod = datafields[self.name][()]
        self.convergence = datafields["Convergence"][()]
        self.l2_precision = datafields["L2gpPrecision"][()]
        self.l2_value = datafields["L2gpValue"][()]
        self.precision = datafields[f"{self.name}Precision"][()]
        self.quality = datafields["Quality"][()]
        self.status = datafields["Status"][()]

    def get_geoloc(self, geolocfields):
        """Method to get all data from geo-location fields

        Helper method to extract the necessary data from the
        'Geolocation Fields' in the MLS file

        Args:
            geolocfields: dataset with geolocation fields
        """
        self.lat = geolocfields["Latitude"][()]
        self.lon = geolocfields["Longitude"][()]
        self.p_grid = geolocfields["Pressure"][()]
        self.time = geolocfields["Time"][()]
        self.dt = make_datetime(self.time)
