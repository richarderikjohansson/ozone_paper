from ozone.io import get_home_data, get_datadir
from scipy.interpolate import interp1d
import numpy as np


def calc_pottemp(t, p):
    kap = 0.286

    pot = t * (1e5 / p) ** kap
    return pot


def add_theta(data, which):
    match which:
        case "MIRA2":
            ptarget = np.load(get_datadir() / "m2pres.npz")["pressure"] * 1e2
            logtgt = np.log(ptarget)
            for val in data.values():
                tmeas = val["tmeas"]
                pmeas = val["pmeas"]
                logsrc = np.log(pmeas)

                interp = interp1d(logsrc, tmeas, bounds_error=False)
                t = interp(logtgt)
                theta = calc_pottemp(t, ptarget)
                val["theta"] = theta

        case "MLSO3":
            dmpdata = np.load(ddir / "dmpdata.npy", allow_pickle=True).item()
            ptarget = np.load(get_datadir() / "m2pres.npz")["pressure"]
            logtgt = np.log(ptarget)
            for dt, val in dmpdata.items():
                if dt in data:
                    dval = data[dt]
                    dval["theta_interp"] = val["theta_interp"]

        case "MLSN2O":
            dmpdata = np.load(ddir / "dmpdata.npy", allow_pickle=True).item()
            ptarget = np.load(get_datadir() / "m2pres.npz")["pressure"]
            logtgt = np.log(ptarget)
            for dt, val in dmpdata.items():
                if dt in data:
                    dval = data[dt]
                    n2oval = dval["N2O"]
                    precision = dval["precision"]
                    psource = dval["pressure"]
                    logsrc = np.log(psource)

                    interpn2o = interp1d(logsrc, n2oval, bounds_error=False)
                    interppres = interp1d(logsrc, precision, bounds_error=False)

                    dval["N2O_interp"] = interpn2o(logtgt)
                    dval["precision_interp"] = interppres(logtgt)
                    dval["p_interp"] = ptarget
                    dval["theta_interp"] = val["theta_interp"]


ddir = get_home_data()
m2O3 = np.load(ddir / "MIRA2_v3_screened_matching.npy", allow_pickle=True).item()
mlsN2O = np.load(ddir / "MLS_N2O_screened_matching.npy", allow_pickle=True).item()
mlsO3 = np.load(ddir / "MLS_O3_screened_matching.npy", allow_pickle=True).item()

add_theta(data=m2O3, which="MIRA2")
add_theta(data=mlsO3, which="MLSO3")
add_theta(data=mlsN2O, which="MLSN2O")
