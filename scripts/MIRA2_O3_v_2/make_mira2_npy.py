from h5py import File
from pathlib import Path
from astropy.time import Time
import numpy as np
from ozone.mira2 import calculate_mr


RET = "MIRA2_O3_v_2"
MEASURE = "mira2_data"
START = "mjd_start"
END = "mjd_end"
MEAN = "mean_mjd"


def find_files():
    home = Path().home()
    root = home / "Data/op"
    gen = root.rglob(pattern="*.hdf5")
    files = list()

    for file in gen:
        with File(file, "r") as handler:
            if RET in handler.keys():
                files.append(file)

    return files


def make_dict(files):
    dct = dict()
    for file in files:
        with File(file, "r") as handler:
            ret = handler[RET]
            measure = handler[MEASURE]
            dt, meastime = make_key(measure=measure)
            convergence = ret.attrs["convergence"]
            mr = calculate_mr(retrieval=ret)

            dct[dt] = {
                "file": np.array([file]),
                "opacity": measure["opacity"][()],
                "transmission": measure["transmission"][()],
                "pmeas": measure["p_grid"][()],
                "zmeas": measure["z_field"][()],
                "tmeas": measure["t_field"][()],
                "meastime": meastime,
                "yf": ret["yf"][()],
                "y": ret["y"][()],
                "residual": ret["y"][()] - ret["yf"][()],
                "f": ret["f_backend"][()],
                "avk": ret["avk"][()][0:41, 0:41],
                "mr": mr,
                "pgrid": ret["p_grid"][()],
                "zgrid": ret["z_field"][()][:, 0, 0],
                "eo": ret["retrieval_eo"][()][0:41],
                "ss": ret["retrieval_ss"][()][0:41],
                "x": ret["x"][()][0:41],
                "apriori": ret["vmr_field"][()][0, :, 0, 0],
                "convergence": convergence,
            }
    dts = [dt for dt in dct.keys()]
    dts_sorted = sorted(dts)
    dct_sorted = {k: dct[k] for k in dts_sorted}
    return dct_sorted


def make_key(measure):
    start = measure[START][()]
    end = measure[END][()]
    mean = measure[MEAN][()]
    meastime = end - start
    dt = Time(mean, format="mjd").to_datetime()
    meastime = Time(meastime, format="mjd")
    return dt, meastime


files = find_files()
data = make_dict(files)
np.save(file="MIRA2_O3_v_2.npy", arr=data, allow_pickle=True)
