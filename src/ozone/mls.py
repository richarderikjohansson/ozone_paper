from pathlib import Path
import h5py
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
from .io import exportdir
from .logger import get_logger
from .utils import fill_nans
from haversine import haversine, Unit


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

    def __init__(self, root: str, make: bool, radii: int):
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
        self.radii = radii
        self.logger = get_logger()
        self.find_mls()

        if make:
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
        umlsdct = {}
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
                        self.lat[i] >= -90
                        and self.lat[i] <= 90
                        and self.lon[i] >= -180
                        and self.lon[i] <= 180
                    ):
                        mls = (self.lat[i], self.lon[i])
                        haversinedist = haversine(
                            self.loc,
                            mls,
                            unit=Unit.KILOMETERS,
                        )
                    else:
                        continue

                    if haversinedist <= self.radii:
                        umlsdct[self.dt[i]] = {
                            f"{self.name}": self.prod[i],
                            "convergence": self.convergence[i],
                            "l2precision": self.l2_precision[i],
                            "l2value": self.l2_value[i],
                            "precision": self.l2_precision[i],
                            "quality": self.quality[i],
                            "status": self.status[i],
                            "lat": self.lat[i],
                            "lon": self.lon[i],
                            "pressure": self.p_grid,
                            "time": self.time[i],
                        }

        sorted_keys = sorted(umlsdct.keys())
        mlsdct = {key: umlsdct[key] for key in sorted_keys}
        sdict = fill_nans(mlsdct)
        savepath = exportdir() / f"{self.name}_{self.radii}km.npy"

        np.save(savepath.resolve(), sdict, allow_pickle=True)
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
