import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from datetime import time


def read_data(file):
    data = np.load(file, allow_pickle=True).item()
    if "mira" in file:
        return get_period(data, period="day")
    else:
        return data


def get_period(data, period="day") -> dict:
    match period:
        case "day":
            lower = time(hour=10, minute=0, second=0)
            upper = time(hour=14, minute=0, second=0)
        case "night":
            lower = time(hour=0, minute=0, second=0)
            upper = time(hour=4, minute=0, second=0)
    dct = {}
    for dt, val in data.items():
        t = dt.time()
        if lower <= t <= upper:
            dct[dt] = val
    return dct


def get_m2_AKS(mls, mira2):
    m2_dt = [dt for dt in mira2.keys()]
    m2_date = np.array([dt.date() for dt in mira2.keys()])
    mls_dt = np.array([dt for dt in mls.keys()])
    mls_date = np.array([dt.date() for dt in mls.keys()])

    aks = dict()
    mls_matching = dict()
    mira2_matching = dict()

    assert len(mls_dt) == len(mls_date)

    for d, mlsdt in zip(mls_date, mls_dt):
        if d in m2_date:
            idx = np.where(m2_date == d)[0]
            dt = m2_dt[idx[0]]
            aks[mlsdt] = mira2[dt]["avk"]
            mls_matching[mlsdt] = mls[mlsdt]

            for i in idx:
                dt = m2_dt[i]
                mira2_matching[dt] = mira2[dt]

    return aks, mls_matching, mira2_matching


mira2 = read_data(
    file="/home/ric/Software/github/op/scripts/timeseries/data/mira2_screened.npy"
)
mls = read_data(file="/home/ric/Downloads/MLS_O3_day_reduced.npy")

fig = plt.figure(figsize=(10, 6))
gs = GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
for i, dt in enumerate(mls.keys()):
    d = dt.date()
    t = dt.time()
    s = (t.hour * 60 + t.minute) * 60 + t.second
    if i == 0:
        ax.scatter(d, s, color="black", s=4, label="MLS")
    else:
        ax.scatter(d, s, color="black", s=4)


for i, dt in enumerate(mira2.keys()):
    d = dt.date()
    t = dt.time()
    s = (t.hour * 60 + t.minute) * 60 + t.second
    if i == 0:
        ax.scatter(d, s, color="tomato", s=4, label="MIRA2")
    else:
        ax.scatter(d, s, color="tomato", s=4)

ax.legend()
ax.set_ylabel("Time")
ax.set_ylabel("Date")
fig.savefig("m2_mls_measmatch.pdf")
plt.close()

aks, mls_matching, mira2_matching = get_m2_AKS(mls=mls, mira2=mira2)
np.save("data/matching/avks.npy", aks, allow_pickle=True)
np.save("data/matching/mls_matching.npy", mls_matching, allow_pickle=True)
np.save("data/matching/mira2_matching.npy", mira2_matching, allow_pickle=True)
