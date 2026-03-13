import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d
from ozone.io import get_downloadsdir, get_datadir
from datetime import date

mlsO3 = np.load("MLS_O3_matched_inside.npy", allow_pickle=True).item()
mlsN2O = np.load("MLS_N2O_matched_inside.npy", allow_pickle=True).item()
m2O3 = np.load("MIRA2_matched_w_theta.npy", allow_pickle=True).item()
mlsO3_tracer = np.load(
    get_downloadsdir() / "O3_tracers_screened_matched.npy", allow_pickle=True
).item()
m2pres = np.load(get_datadir() / "m2pres.npz")["pressure"]
mlspres = np.load(get_downloadsdir() / "pressure.npz")["pressure"]

check = date(2020, 2, 14)

for k in mlsO3:
    d = k.date()
    if d == check:
        mlsk = k
        break

for k in m2O3:
    d = k.date()
    if d == check:
        m2k = k
        break

m2theta = m2O3[m2k]["theta"]
m2p = m2O3[m2k]["pgrid"]
m2prodtheta = m2O3[m2k]["x_phys"]

mlstheta = mlsO3[mlsk]["theta"]
mlsp = mlsO3[mlsk]["pressure"]
mlsprodtheta = mlsO3_tracer[mlsk]["O3_interp"] * 1e6

thet = np.zeros_like(mlstheta)
for i, t in enumerate(mlstheta):
    if t > 100000:
        thet[i] = np.nan
    else:
        thet[i] = t


fig = plt.figure(figsize=(8, 13))
gs = GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])

ax.set_yscale("log")
ax.plot(m2prodtheta, m2theta)
ax.plot(mlsprodtheta, thet)
fig.savefig("mls_m2_theta.pdf")

fig = plt.figure(figsize=(8, 13))
gs = GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])

ax.set_yscale("log")
ax.plot(m2prodtheta, m2theta)
ax.plot(mlsprodtheta, thet)
fig.savefig("mls_m2_theta.pdf")

psrc = np.log(mlspres)
ptgt = np.log(m2pres)

interptheta = interp1d(psrc, mlstheta, kind="linear", fill_value="extrapolate")
interpO3 = interp1d(psrc, mlsprodtheta)

mlstheta_new = interptheta(ptgt)
mlsO3_new = interpO3(ptgt)

thet = np.zeros_like(mlstheta_new)
for i, t in enumerate(mlstheta_new):
    if t > 100000:
        thet[i] = np.nan
    else:
        thet[i] = t

fig = plt.figure(figsize=(8, 13))
gs = GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])

ax.set_yscale("log")
ax.plot(m2prodtheta, m2theta)
ax.plot(mlsO3_new, thet)
fig.savefig("mls_m2_theta_new.pdf")
