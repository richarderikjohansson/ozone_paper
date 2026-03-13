import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

mls = np.load("MLS_with_smooth.npy", allow_pickle=True).item()
m2 = np.load("MIRA2_O3_v_2.npy", allow_pickle=True).item()
gs = GridSpec(1, 1)

# fig = go.Figure(
#    layout=dict(
#        template="plotly_white",
#        width=800,
#        height=100,
#    )
# )
#
# for v in mls.values():
#    print(v["O3new"])
#    print(v["pnew"])
#    fig.add_trace(go.Scatter(x=v["O3smooth"], y=v["pnew"]))
#
# fig.update_yaxes(
#    autorange="reversed",
#    visible=True,
#    linecolor="black",
#    linewidth=2,
#    type="log",
#    exponentformat="power",
# )
#
# fig.show()

fig = plt.figure(figsize=(6, 10))
ax = fig.add_subplot(gs[0, 0])
ax.invert_yaxis()
ax.set_yscale("log")
for v in m2.values():
    p = v["pgrid"] / 1e2
    prod = v["apriori"] * v["x"] * 1e6
    mr = v["mr"]
    ax.plot(mr, p)

fig.savefig("mira2_mr.pdf")
plt.close()

fig = plt.figure(figsize=(6, 10))
ax = fig.add_subplot(gs[0, 0])
for v in m2.values():
    f = v["f"]
    res = v["residual"]
    ax.plot(f, res)

fig.savefig("mira2_res.pdf")
plt.close()
