from .io import get_datadir, get_exportdir
from .utils import fill_nans, find_downloads
from .logger import get_logger
import numpy as np
import yaml
from datetime import datetime


class DataScreener:
    def __init__(self, dataset, filename):
        self.ddir = get_datadir()
        self.edir = get_exportdir()
        self.dataset = dataset
        self.filename = filename

        self.find_dataset()
        self.find_screen_file()
        self.read_data()

    def find_dataset(self):
        files = [file for file in self.edir.rglob(pattern=f"{self.dataset}*")]
        files = np.array(files)

        assert len(files) > 0, (
            "Check dataset name and that it exitstsin $HOME/.cache/m2exports"
        )

        for file in files:
            if "meta" in file.name:
                self.metadata_fp = file
            else:
                self.dataset_fp = file

    def find_screen_file(self):
        self.screen_fp = self.ddir / f"{self.dataset}.yaml"

    def read_data(self):
        self.meta = np.load(self.metadata_fp, allow_pickle=True).item()
        self.data = np.load(self.dataset_fp, allow_pickle=True).item()

        with open(self.screen_fp, "r") as fh:
            self.screen = yaml.load(fh, Loader=yaml.SafeLoader)

        assert self.meta["product"] == self.screen["dataset"]
        return self.data, self.meta, self.screen


class MLSScreener:
    def __init__(self, data, meta, screen, logger, winter):
        self.meta = meta
        self.data = data
        self.screen = screen
        self.logger = logger
        self.winter = winter
        self.get_data()

    def get_data(self):
        self.dt = np.array([dt for dt in self.data.keys()])
        self.precision = np.array([data["precision"] for data in self.data.values()])
        self.status = np.array([data["status"] for data in self.data.values()])
        self.quality = np.array([data["quality"] for data in self.data.values()])
        self.convergence = np.array(
            [data["convergence"] for data in self.data.values()]
        )
        self.pressure = np.array([data["pressure"] for data in self.data.values()])

    def _screen_status(self):
        status = self.screen["status"]
        if status == "not_odd":
            self.status_mask = self.status % 2 == 0

        if status == "equal_zero":
            self.status_mask = self.status == 0

    def _screen_quality(self):
        quality = self.screen["quality"]
        self.quality_mask = self.quality > quality

    def _screen_convergence(self):
        convergence = self.screen["convergence"]
        self.convergence_mask = self.convergence < convergence

    def _screen_precision(self):
        msk = []
        pmin = self.screen["pmin"]
        pmax = self.screen["pmax"]

        for i, p in enumerate(self.pressure):
            imax = np.where(p < pmax)[0]
            imin = np.where(p < pmin)[0]

            if len(imax) > 0 and len(imin) > 0:
                j = imax[0]
                k = imin[0]
                p = self.precision[i]
                ploc = p[j:k]
                mask = ploc > self.screen["precision"]
                if False in mask:
                    msk.append(False)
                else:
                    msk.append(True)
            else:
                msk.append(False)

        self.precision_mask = np.array(msk)

    def _screen_winter(self):
        start = datetime(2019, 10, 1, 0, 0, 0)
        end = datetime(2020, 5, 1, 0, 0, 0)
        dts = np.array([dt for dt in self.data.keys()])
        self.winter_mask = (dts >= start)&(dts <=end)

    def save_screened_data(self, filename):
        if self.winter:
            self._screen_winter()
        else:
            self.winter_mask = np.array([True for dt in self.data.keys()])
        self._screen_status()
        self._screen_quality()
        self._screen_convergence()
        self._screen_precision()

        combined_mask = (
            self.status_mask
            & self.quality_mask
            & self.convergence_mask
            & self.precision_mask
            & self.winter_mask
        )
        screened_dts = self.dt[combined_mask]
        mdict = {dt: self.data[dt] for dt in screened_dts}
        sdict = fill_nans(mdict=mdict)
        savepath = find_downloads() / f"{filename}.npy"
        product = self.meta["product"]
        np.save(savepath, sdict, allow_pickle=True)
        self.logger.info(f"Screened {product} file saved in {savepath}")


class MIRA2Screener:
    def __init__(self, data, meta, screen, logger):
        self.meta = meta
        self.data = data
        self.screen = screen
        self.logger = logger
