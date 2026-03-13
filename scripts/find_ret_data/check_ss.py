import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def normalize_array(arr):
    norm_arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr)) + 0.1
    return norm_arr


def sigmoid(arr):
    return 1 / (1 + np.exp(-0.02 * arr))


def softmax(arr):
    exp = np.exp(arr)
    return np.exp(arr) / exp.sum()


def siglog(p, w_min, p0, k):
    # w_min = 0.1
    # p0 = 10.0
    # k = 4.0

    weights = w_min + (1 - w_min) / (1 + np.exp(-k * (np.log10(p0) - np.log10(p))))
    return weights


with open("MIRA2_O3_v_1.json", "r") as fp:
    data = json.load(fp)

m2 = np.load("mira2_matching.npy", allow_pickle=True).item()
pgrid = np.load("pgrid.npy", allow_pickle=True) / 1e2
sx = data["Sx_diag_O3"]
prod = []

for ds in m2.values():
    p = ds["x"] * ds["apriori"] * 1e6
    prod.append(p)


pgrid = [
    1.000e3,
    8.254e2,
    6.813e2,
    5.623e2,
    4.642e2,
    3.831e2,
    3.162e2,
    2.610e2,
    2.154e2,
    1.778e2,
    1.468e2,
    1.212e2,
    1.000e2,
    8.254e1,
    6.813e1,
    5.623e1,
    4.642e1,
    3.831e1,
    3.162e1,
    2.610e1,
    2.154e1,
    1.778e1,
    1.468e1,
    1.212e1,
    1.000e1,
    8.254e0,
    6.813e0,
    5.623e0,
    4.642e0,
    3.831e0,
    3.162e0,
    2.610e0,
    2.154e0,
    1.778e0,
    1.468e0,
    1.212e0,
    1.000e0,
    6.813e-1,
    4.642e-1,
    3.162e-1,
    2.154e-1,
    1.468e-1,
    1.000e-1,
    4.642e-2,
    2.154e-2,
    1.000e-2,
]
w = siglog(pgrid, w_min=0.1, p0=10, k=3)
print(w)
fig = plt.figure(figsize=(10, 6))
gs = GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.set_yscale("log")
ax.invert_yaxis()

ax.plot(sx, pgrid, color="red", linewidth=2)
ax.plot(w, pgrid)
ax.set_xlabel("%")
ax.set_ylabel("p [hPa]")
ax.minorticks_on()
ax.grid(visible=True, which="both", alpha=0.3)
fig.savefig("MIRA2_O3_v_2_Sx.pdf")
