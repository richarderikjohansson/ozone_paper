import plotly.graph_objects as go
import numpy as np
from pathlib import Path

data = np.load("month_means.npz")
indecies = range(1, 13)
p = np.load("pf.npy")
keys = [
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
]

dct = {i: data[k] for i, k in zip(indecies, keys)}

fig = go.Figure(layout=dict(template="plotly_white", height=1100, width=700))
fig.update_yaxes(
    type="log",
    autorange="reversed",
    title="Pressure [hPa]",
    tickfont=dict(size=20),
    exponentformat="power",
    showexponent="all",
    ticklen=8,
    minor=dict(ticks="inside", ticklen=3, showgrid=True),
)
fig.update_xaxes(range=[-0.5, 7])
fig.update_xaxes(title="VMR [ppm]", tickfont=dict(size=17))
fig.update_layout(
    xaxis_title_font=dict(size=22),
    yaxis_title_font=dict(size=22),
    xaxis=dict(tickfont=dict(size=20)),
    yaxis=dict(tickfont=dict(size=20)),
    legend=dict(font=dict(size=16)),
)
for k, apri in zip(keys, dct.values()):
    fig.add_trace(
        go.Scatter(
            x=apri * 1e6,
            y=p,
            name=k.capitalize(),
        )
    )

fig.write_html("aprioris.html")
