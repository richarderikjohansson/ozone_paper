from h5py import File
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.gridspec import GridSpec
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def read_file(path):
    with File(path, "r") as handler:
        data = handler["mira2_data"]
        dct = {k: v[()] for k, v in data.items()}
    return dct


def find_files():
    home = Path().home()
    root = home / "Data/op/"
    generator = root.rglob(pattern="*.hdf5")
    files = [file for file in generator]
    return files


def estimate_noise_std(x, window=25):
    residuals = []

    for i in range(len(x) - window + 1):
        w = x[i : i + window]
        t = np.arange(window)

        # Linear fit to local signal
        p = np.polyfit(t, w, 1)
        trend = np.polyval(p, t)

        residuals.extend(w - trend)

    residuals = np.array(residuals)

    # Robust estimate
    sigma = 1.4826 * np.median(np.abs(residuals))
    return sigma, residuals


files = find_files()
data = read_file(path=files[0])

fig = plt.figure(figsize=(15, 8))
gs = GridSpec(1, 2)
left = fig.add_subplot(gs[0, 0])
right = fig.add_subplot(gs[0, 1])

f = data["f"]
y = data["y"]
for file in files:
    data = read_file(file)
    y = data["y"]
    sigma, res = estimate_noise_std(y)
    if sigma < 0.1:
        fsub = data["f"][0:75]
        ysub = y[0:75]
        left.plot(fsub, ysub)
        right.hist(ysub, bins=20)

        fig.savefig("test.pdf")
        print(np.var(ysub))
        print(sigma)
        break
