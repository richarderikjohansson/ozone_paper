from ozone.io import get_home_data
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from ozone._const import COLORS
import numpy as np
from datetime import date

m2lossp = get_home_data() / "m2loss.npy"
mlslossp = get_home_data() / "mlsloss.npy"

m2loss = np.load(m2lossp, allow_pickle=True).item()
mlsloss = np.load(mlslossp, allow_pickle=True).item()

isokeys = [k for k in m2loss.keys()]

theta_upper = round(np.median(m2loss[isokeys[0]]["theta"]))
theta_bottom = round(np.median(m2loss[isokeys[-1]]["theta"]))


fig = plt.figure(figsize=(13, 8))
gs = GridSpec(2, 1)

upper = fig.add_subplot(gs[0, 0])
bottom = fig.add_subplot(gs[1, 0], sharex=upper)

ylims = (-3.5, 2)
xlims = (date(2019, 10, 1), date(2020, 4, 30))

for ax in fig.axes:
    ax.set_ylim(ylims)
    ax.set_xlim(xlims)
    ax.set_ylabel(r"$\Delta O_3$")
    # ax.minorticks_on()
    ax.grid(which="both", alpha=0.1)
    ax.hlines(y=0, xmin=xlims[0], xmax=xlims[-1], color=COLORS.overlay, alpha=0.5)


upper.plot(
    mlsloss[isokeys[0]]["dates"],
    mlsloss[isokeys[0]]["loss"] * 1e6,
    color=COLORS.mantle,
    label=r"$\Delta O_3 \pm \sigma$ (MLS)",
    marker="o",
)
upper.fill_between(
    x=mlsloss[isokeys[0]]["dates"],
    y1=(mlsloss[isokeys[0]]["loss"] + mlsloss[isokeys[0]]["loss_err"]) * 1e6,
    y2=(mlsloss[isokeys[0]]["loss"] - mlsloss[isokeys[0]]["loss_err"]) * 1e6,
    color=COLORS.mantle,
    alpha=0.1,
)
upper.plot(
    m2loss[isokeys[0]]["dates"],
    m2loss[isokeys[0]]["loss"] * 1e6,
    marker="o",
    color=COLORS.red,
    label=r"$\Delta O_3 \pm \sigma$ (MIRA2)",
)
upper.fill_between(
    x=m2loss[isokeys[0]]["dates"],
    y1=(m2loss[isokeys[0]]["loss"] + m2loss[isokeys[0]]["loss_err"]) * 1e6,
    y2=(m2loss[isokeys[0]]["loss"] - m2loss[isokeys[0]]["loss_err"]) * 1e6,
    color=COLORS.red,
    alpha=0.1,
)
upper.set_title(rf"~{theta_upper} K ({isokeys[0]} ppbv $N_2O$)")
upper.legend()

bottom.plot(
    mlsloss[isokeys[-1]]["dates"],
    mlsloss[isokeys[-1]]["loss"] * 1e6,
    color=COLORS.mantle,
    marker="o",
    label=r"$\Delta O_3 \pm \sigma$ (MLS)",
)
bottom.fill_between(
    x=mlsloss[isokeys[-1]]["dates"],
    y1=(mlsloss[isokeys[-1]]["loss"] + mlsloss[isokeys[-1]]["loss_err"]) * 1e6,
    y2=(mlsloss[isokeys[-1]]["loss"] - mlsloss[isokeys[-1]]["loss_err"]) * 1e6,
    color=COLORS.mantle,
    alpha=0.1,
)
bottom.plot(
    m2loss[isokeys[-1]]["dates"],
    m2loss[isokeys[-1]]["loss"] * 1e6,
    color=COLORS.red,
    marker="o",
    label=r"$\Delta O_3 \pm \sigma$ (MIRA2)",
)
bottom.fill_between(
    x=m2loss[isokeys[-1]]["dates"],
    y1=(m2loss[isokeys[-1]]["loss"] + m2loss[isokeys[-1]]["loss_err"]) * 1e6,
    y2=(m2loss[isokeys[-1]]["loss"] - m2loss[isokeys[-1]]["loss_err"]) * 1e6,
    color=COLORS.red,
    alpha=0.1,
)
bottom.set_title(rf"~{theta_bottom} K ({isokeys[-1]} ppbv $N_2O$)")
bottom.legend()


bottom.set_xlabel("Date")


fig.savefig("loss_kir_2020.pdf")
