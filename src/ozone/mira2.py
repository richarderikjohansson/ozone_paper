from pathlib import Path
import h5py
from tqdm import tqdm
import numpy as np
from datetime import datetime
from .io import get_exportdir
from .logger import get_logger
from .utils import fill_nans


def make_datetime(measure: h5py._hl.group.Group) -> datetime:
    """Function to make datetime objects

    This function takes the measurement dataset
    and parses the start and end times of the
    measurements. It will return the datetime
    in the middle of the measurement

    Args:
        measure: dataset containing measurement data

    Returns:
        middle of the measurement datetime
    """
    start_year, end_year = (
        measure["start_year"][()].decode(),
        measure["end_year"][()].decode(),
    )
    start_month, end_month = (
        measure["start_month"][()].decode(),
        measure["end_month"][()].decode(),
    )
    start_day, end_day = (
        measure["start_day"][()].decode(),
        measure["end_day"][()].decode(),
    )
    start_hour, end_hour = (
        measure["start_hour"][()].decode(),
        measure["end_hour"][()].decode(),
    )
    start_min, end_min = (
        measure["start_min"][()].decode(),
        measure["end_min"][()].decode(),
    )
    start_sec, end_sec = (
        measure["start_sec"][()].decode(),
        measure["end_sec"][()].decode(),
    )

    start = datetime.strptime(
        f"{start_year}{start_month}{start_day}{start_hour}{start_min}{start_sec}",
        "%Y%m%d%H%M%S",
    )
    end = datetime.strptime(
        f"{end_year}{end_month}{end_day}{end_hour}{end_min}{end_sec}", "%Y%m%d%H%M%S"
    )
    delta = end - start
    mid = start + delta / 2
    return mid


def calculate_mr(retrieval: h5py._hl.group.Group) -> np.ndarray:
    """Function to calculate the measurement response

    This function takes the retrieval dataset and
    calculates the measurement response from the
    averaging kernel matrix by summing the AK's
    rows

    Args:
        retrieval: Dataset containing retrieval data

    Returns:
        Measurement response
    """
    mr = []
    for row in retrieval["avk"][()][0:41, 0:41]:
        mr.append(sum(row))
    return np.array(mr)


class MIRA2FindAndMake:
    """
    Class to find all the MIRA2 files, and to create
    two new files containing relevant measurement and
    retrieval data.
    """

    def __init__(self, root: str, make: bool):
        """Init constructor

        Args:
            root: Path to the directory with MIRA2 files
            make: Boolean if files should be created
        """
        self.KEY = "MIRA2_O3_v_1"
        self.root = Path(root).resolve()
        self.find_mira2()
        self.logger = get_logger()

        if make:
            self.makeproducts()

    def find_mira2(self):
        """Method to locate all MIRA2 files

        This method recursively searches the root path
        given for .hdf5 files. It also checks that the
        file is containing a retrieval, and that this
        retrieval have converged from the metadata written
        to the file.
        """
        self.files = self.root.rglob(pattern="*.hdf5")
        retfiles = []

        for file in tqdm(self.files, desc="Finding files with retrieval"):
            with h5py.File(file, "r") as fh:
                if self.KEY in fh.keys():
                    retfiles.append(file.resolve())

        retfiles = np.array(sorted(retfiles))
        self.retfiles = retfiles

    def makeproducts(self):
        """Method to create new files

        This method is extracting all relevant data from the
        measurement and retrieval for the files found with
        retrieval data. It then places this data in two
        dictionaries with the datetime as a top level key
        for each set of data.

        It finally saves these two dictionaries in the export
        directory under 'measure.npy' and 'retrieval.npy' for
        the measurement and retrieval data respectively
        """
        edir = get_exportdir()
        mdict = {}

        for file in tqdm(self.retfiles, desc="Extracting products"):
            with h5py.File(file, "r") as f:
                measure = f["mira2_data"]
                retrieval = f[self.KEY]

                dt = make_datetime(measure)
                mdict[dt] = {
                    "opacity": measure["opacity"][()],
                    "transmission": measure["transmission"][()],
                    "pmeas": measure["p_grid"][()],
                    "zmeas": measure["z_field"][()],
                    "tmeas": measure["t_field"][()],
                    "meastime": measure["meas_duration"][()],
                    "yf": retrieval["yf"][()],
                    "y": retrieval["y"][()],
                    "residual": retrieval["y"][()] - retrieval["yf"][()],
                    "f": retrieval["f_backend"][()],
                    "avk": retrieval["avk"][()][0:41, 0:41],
                    "mr": calculate_mr(retrieval),
                    "pgrid": retrieval["p_grid"][()],
                    "zgrid": retrieval["z_field"][()][:, 0, 0],
                    "eo": retrieval["retrieval_eo"][()][0:41],
                    "ss": retrieval["retrieval_ss"][()][0:41],
                    "x": retrieval["x"][()][0:41],
                    "apriori": retrieval["vmr_field"][()][0, :, 0, 0]
                }

        sdict = fill_nans(mdict)
        savepath = edir / "mira2.npy"
        metapath = edir / "mira2.meta.npy"
        mdict = {
            "product": "mira2",
            "make_date": datetime.now(),
            "sources": self.retfiles
        }

        np.save(savepath, sdict, allow_pickle=True)
        np.save(metapath, mdict, allow_pickle=True)
        self.logger.info(f"Saved measurement and retrieval data into {savepath}")
