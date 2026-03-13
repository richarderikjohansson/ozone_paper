import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from ozone.analysis import interp_mls
from cmcrameri import cm


def read_data(file):
    if "pgrid" in file.name or "apriori" in file.name:
        data = np.load(file, allow_pickle=True)
        return data
    else:
        data = np.load(file, allow_pickle=True).item()
        return data


def find_files():
    cwd = Path(__file__).parent
    matching = cwd / "data/matching_night"
    files = matching.glob("*.npy")

    for file in files:
        if "avk" in file.name:
            avk = file
        if "mira2" in file.name:
            mira2 = file
        if "mls" in file.name:
            mls = file
        if "pgrid" in file.name:
            pgrid = file
        if "apriori" in file.name:
            apriori = file

    return avk, mira2, mls, pgrid, apriori


def save_mls(mls):
    cwd = Path(__file__).parent
    savedir = cwd / "data/smooth"
    np.save(savedir / "mls_smooth_night.npy", mls, allow_pickle=True)


def get_bounds(mira2, mls):
    scaler = 1e6
    m2O3 = [val["apriori"] * val["x"] for val in mira2.values()]
    mlsO3 = [val["O3smooth"] for val in mls.values()]
    mlstmpmx = []
    m2tmpmx = []

    for arr in mlsO3:
        mlstmpmx.append(max(arr))

    mls_mx = max(mlstmpmx) * scaler

    for arr in m2O3:
        m2tmpmx.append(max(arr))

    m2_mx = max(m2tmpmx) * scaler
    mx = max(mls_mx, m2_mx)

    bounds = np.linspace(0, mx, 12)
    return bounds


def get_mls_mean_pressure(mls):
    pres = np.array([val["pressure"] for val in mls.values()])
    assert np.all(pres)
    mpres = pres.mean(axis=0)
    return mpres


avk, mira2, mls, pgrid, apriori = find_files()

avk = read_data(avk)
m2 = read_data(mira2)
mls = read_data(mls)
pgrid = read_data(pgrid)
apriori = read_data(apriori)

mls = interp_mls(avk=avk, mls=mls, ptarget=pgrid / 1e2, apriori=apriori)
mlspres = get_mls_mean_pressure(mls)
save_mls(mls)
bounds = get_bounds(m2, mls)

# form data
m2O3 = np.array([(val["apriori"] * val["x"]) * 1e6 for val in m2.values()])
m2dt = [dt for dt in m2.keys()]
mls_smooth = np.array([val["O3smooth"] * 1e6 for val in mls.values()])
mls_org = np.array([val["O3new"] * 1e6 for val in mls.values()])
mlsdt = [dt for dt in mls.keys()]

# figure
fig = plt.figure(figsize=(12, 10))
gs = GridSpec(3, 1)
top = fig.add_subplot(gs[0, 0])  # m2
middle = fig.add_subplot(gs[1, 0])  # mls smooth
bottom = fig.add_subplot(gs[2, 0])  # mls org


for i, ax in enumerate(fig.axes):
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_ylabel("Pressure [hPa]", fontsize=13)
    ax.set_ylim([100, 1e-1])

    match i:
        case 0:  # m2
            ax.set_title("MIRA2")
            X, Y = np.meshgrid(m2dt, pgrid)
            m2cf = ax.contourf(X, Y / 1e2, m2O3.T, levels=bounds, cmap=cm.vik)
        case 1:  # mls smooth
            ax.set_title("MLS (AK convolve)")
            X, Y = np.meshgrid(mlsdt, pgrid)
            mlscf_smooth = ax.contourf(
                X, Y / 1e2, mls_smooth.T, levels=bounds, cmap=cm.vik
            )
        case 2:
            ax.set_title("MLS org")
            mlscf_org = ax.contourf(X, Y / 1e2, mls_org.T, levels=bounds, cmap=cm.vik)
            ax.set_xlabel("Date (UTC)", fontsize=13)

fig.colorbar(m2cf, ax=top)
fig.colorbar(mlscf_smooth, ax=middle)
fig.colorbar(mlscf_org, ax=bottom)
fig.savefig("ts_interp.pdf")
plt.close()

fig = plt.figure(figsize=(12, 3))
gs = GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.pcolormesh(m2O3.transpose())
fig.savefig("pcolormesh.pdf")
