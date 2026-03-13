import numpy as np
from datetime import datetime, timedelta
from ozone.analysis import match_measurements, interp_mls


def combine_day_and_night(day, night):
    daydt = np.array([dt for dt in day.keys()])
    nightdt = np.array([dt for dt in night.keys()])
    dct = dict()
    dts = np.concatenate((daydt, nightdt))
    dts_sorted = sorted(dts)
    for dt in dts_sorted:
        try:
            data = day[dt]
        except KeyError:
            data = night[dt]

        dct[dt] = data

    return dct


day = np.load("MLS_O3_day_reduced.npy", allow_pickle=True).item()
night = np.load("MLS_O3_night_reduced.npy", allow_pickle=True).item()
mls = combine_day_and_night(day, night)
mira2 = np.load("MIRA2_O3_v_2.npy", allow_pickle=True).item()
pgrid = np.load("pgrid.npy", allow_pickle=True)
apriori = np.load("apriori.npy", allow_pickle=True)
avks, mls_match, mira2_match = match_measurements(mira2=mira2, mls=mls)
mls = interp_mls(avks, mls_match, pgrid, apriori)
np.save("MLS_with_smooth.npy", mls, allow_pickle=True)
