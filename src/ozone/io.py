from pathlib import Path
import h5py
from tqdm import tqdm
import numpy as np
from datetime import datetime


def exportdir():
    filedir = Path(__file__).resolve()
    ddir = filedir.parent / "data"
    edir = ddir / "export"
    if not edir.exists():
        edir.mkdir()

    return edir


def make_datetime(measure):
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

def calculate_mr(retrieval):
    mr = []
    for row in retrieval["avk"][()][0:41, 0:41]:
        mr.append(sum(row))
    return mr


class MIRA2FindAndMake:
    def __init__(self, root: str, make: bool):
        self.KEY = "MIRA2_O3_v_1"
        self.root = Path(root)
        self.find_mira2()

        if make:
            self.makeproducts()

    def find_mira2(self):
        files = self.root.rglob(pattern="*.hdf5")
        retfiles = []

        for file in tqdm(files, desc="Finding files with retrieval"):
            with h5py.File(file, "r") as fh:
                if self.KEY in fh.keys():
                    data = fh[self.KEY]
                    conv = data.attrs["convergence"]
                    if conv == 0.0:
                        retfiles.append(file.resolve())

        retfiles = np.array(sorted(retfiles))
        self.retfiles = retfiles

    def makeproducts(self):
        edir = exportdir()
        mdict = {}
        rdict = {}

        for file in tqdm(self.retfiles, desc="Measurement products"):
            with h5py.File(file, "r") as f:
                measure = f["mira2_data"]
                retrieval = f[self.KEY]

                dt = make_datetime(measure)
                mdict[dt] = {
                    "opacity": measure["opacity"][()],
                    "transmission": measure["transmission"][()],
                    "pgrid": measure["p_grid"][()],
                    "zgrid": measure["z_field"][()],
                    "tgrid": measure["t_field"][()]
                }

        np.save(edir / "measure.npy", mdict, allow_pickle=True)
        del mdict

        for file in tqdm(self.retfiles, desc="Retrieval products"):
            with h5py.File(file, "r") as f:
                measure = f["mira2_data"]
                retrieval = f[self.KEY]
                dt = make_datetime(measure)
                rdict[dt] = {
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

        np.save(edir / "retrieval.npy", rdict, allow_pickle=True)
        del rdict
