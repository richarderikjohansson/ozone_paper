from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cmcrameri.cm as cm
from cycler import cycler
import pyarts

# Number of lines
n_colors = 12

# Sample evenly from the cmcraie colormap
colors = cm.managua(np.linspace(0, 1, n_colors))

# Set as default color cycle
plt.rcParams["axes.prop_cycle"] = cycler(color=colors)


def read_all_data():
    cwd = Path(__file__).parent
    ddir = cwd / "monthly_means"
    generator = ddir.rglob("*.npy")
    out = {i: [] for i in range(1, 13)}
    means = {}
    z = np.loadtxt("era5/z.txt")

    for file in generator:
        data = np.load(file, allow_pickle=True).item()
        for i, v in data.items():
            out[i].append(v[::-1])
        plt.close()

    for i, v in out.items():
        means[i] = np.mean(v, axis=0)

    return means


means = read_all_data()
z = np.loadtxt("era5/pf.txt")[::-1]
fig, ax = plt.subplots(figsize=(6, 10))
ax.set_yscale("log")
ax.invert_yaxis()
for i, v in means.items():
    ax.plot(v, z, label=i)
ax.legend()
fig.savefig("monthly_means.pdf")
vals = [v for v in means.values()]

for xi in means[12]:
    print(xi)

# print(means[1])

np.save("monthly_means.npy", means, allow_pickle=True)
np.savez_compressed(
    "month_means.npz",
    jan=vals[0],
    feb=vals[1],
    mar=vals[2],
    apr=vals[3],
    may=vals[4],
    jun=vals[5],
    jul=vals[6],
    aug=vals[7],
    sep=vals[8],
    oct=vals[9],
    nov=vals[10],
    dec=vals[11],
)
np.save("pf.npy", z)
