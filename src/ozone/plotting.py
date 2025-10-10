import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from .utils import find_downloads
from .io import get_simulationdir, get_screendir
import numpy as np

class Plotting:
    def __init__(self, logger):
        self.logger = logger
        self.ddir = find_downloads()
        self.f0_mira2 = 273.051


    def make_fig01(self, figure):
        filename = get_simulationdir() / f"{figure}.npy"
        figname = self.ddir / f"{figure}.pdf"
        data = np.load(filename, allow_pickle=True).item()
        xy = {
            "MIRA2": (self.f0_mira2-1, 73),
            "R2": (190, 62.3),
            "R3": (229, 57)
        }

        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(1, 1)
        ax = fig.add_subplot(gs[0, 0])

        ax.plot(data["f"]/1e9, data["I"], color="black")
        ax.set_ylabel(r"$T_B$ $[K]$", fontsize=16, labelpad=10)
        ax.set_xlabel(r"$\nu$ $[GHz]$", fontsize=16, labelpad=10)
        ax.tick_params(labelsize=14)
        ax.add_patch(
            Rectangle(
                xy=xy["MIRA2"],
                width=2,
                height=23,
                fill=False,
                edgecolor="red",
                lw=1
            )
        )
        ax.annotate("MIRA2",
                    xy=(self.f0_mira2, 96),
                    ha="center",
                    xytext=(self.f0_mira2, 120),
                    arrowprops=dict(facecolor='black', shrink=0.1, width=0.5))
        ax.add_patch(
            Rectangle(
                xy=xy["R2"],
                width=17,
                height=70,
                fill=False,
                edgecolor="red",
                lw=1
            )
        )
        ax.annotate("MLS/Aura R2",
                    xy=(203, 133),
                    ha="center",
                    xytext=(203, 150),
                    arrowprops=dict(facecolor='black', shrink=0.1, width=0.5))
        ax.add_patch(
            Rectangle(
                xy=xy["R3"],
                width=12,
                height=43,
                fill=False,
                edgecolor="red",
                lw=1
            )
        )
        ax.annotate("MLS/Aura R3",
                    xy=(229+(12/2), 101),
                    ha="center",
                    xytext=(229+(12/2), 120),
                    arrowprops=dict(facecolor='black', shrink=0.1, width=0.5))
        ax.grid(alpha=0.2)
        fig.savefig(figname, transparent=True)
        plt.show()
        plt.close()
        self.logger.info(f"Saved Figure 1 in {figname}")


def dynamic_caller(obj, method_name):
    method = getattr(obj, method_name, None)

    if callable(method):
        return method
    else:
        raise AttributeError(f"Method {method_name} not found")
