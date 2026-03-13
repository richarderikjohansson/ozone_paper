from ozone.io import get_downloadsdir
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from datetime import date, time
from IPython.display import display

dmpdata = np.load(get_downloadsdir() / "dmpdata.npy", allow_pickle=True).item()
lats = []
lons = []
eqls = []
pvs = []
gradpvs = []
spvs = []
timestamps = []

for timestamp, data in dmpdata.items():
    try:
        d = timestamp.date()
        t = timestamp.time()

        # if d == date(2020, 2, 14) and t >= time(19, 26, 10) and t <= time(19, 32, 21):
        isentrope = 475
        if d == date(2020, 1, 18) and t >= time(1, 37, 50) and t <= time(1, 42, 20):
            latitude = data["latitude"]
            longitude = data["longitude"]
            theta = data["theta"]
            pv = data["pv"]
            spv = data["spv"]
            gradpv = data["gradpv"]
            eql = data["eql"]
            lats.append(data["latitude"])
            lons.append(data["longitude"])
            mask = (theta > isentrope) & (theta < 2000)
            print(theta[mask][0])
            try:
                eqls.append(eql[mask][0])
                pvs.append(pv[mask][0])
                gradpvs.append(gradpv[mask][0])
                spvs.append(spv[mask][0])
                timestamps.append(timestamp)
            except IndexError:
                continue

    except AttributeError:
        continue

print(len(pvs))
print(len(timestamps))
print(eqls)
# eqls = eqls[0:-3]
# pvs = pvs[0:-3]
# gradpvs = gradpvs[0:-3]
# lats = lats[0:-3]
# lons = lons[0:-3]

# fig = go.Figure(layout=dict(template="plotly_white", width=1000, height=800))
ids = list(range(len(lats)))
fig = make_subplots(
    rows=3,
    cols=2,
    specs=[
        [{"type": "xy"}, {"type": "geo"}],
        [{"type": "xy"}, {"type": "geo"}],
        [{"type": "xy"}, {"type": "geo"}],
    ],
    subplot_titles=(
        f"Gradient of PV at ~{isentrope} K",
        "MLS track January 2020",
        f"PV at ~{isentrope} K",
    ),
)

# Top-left scatter
fig.add_scatter(
    x=eqls,
    y=gradpvs,
    mode="markers+lines",
    ids=ids,
    name="PV gradient",
    customdata=list(zip(lats, lons, eqls, pvs)),
    hovertemplate="Latitude: %{customdata[0]}<br>Longitude: %{customdata[1]}<br>EQL: %{customdata[2]}<br>PV: %{customdata[3]}<extra></extra>",
    row=1,
    col=1,
)

# Bottom-left scatter
fig.add_scatter(
    x=eqls,
    y=pvs,
    mode="markers+lines",
    name="PV",
    ids=ids,
    customdata=list(zip(lats, lons, eqls)),
    hovertemplate="Latitude: %{customdata[0]}<br>Longitude: %{customdata[1]}<br>EQL: %{customdata[2]}<extra></extra>",
    row=2,
    col=1,
)
fig.add_scatter(
    x=eqls,
    y=spvs,
    mode="markers+lines",
    name="Scaled PV",
    ids=ids,
    customdata=list(zip(lats, lons, eqls)),
    hovertemplate="Latitude: %{customdata[0]}<br>Longitude: %{customdata[1]}<br>EQL: %{customdata[2]}<extra></extra>",
    row=3,
    col=1,
)

# Top-right geoplot
fig.add_trace(
    go.Scattergeo(
        lat=lats,
        lon=lons,
        mode="markers",
        name="MLS footprint",
        ids=ids,
        customdata=list(zip(lats, lons, timestamps, eqls, pvs)),
        hovertemplate="Latitude: %{customdata[0]}<br>Longitude: %{customdata[1]}<br>Timestamp: %{customdata[2]}<br>EQL: %{customdata[3]}<br>PV: %{customdata[4]}<extra></extra>",
    ),
    row=2,
    col=2,
)
fig.add_trace(
    go.Scattergeo(lat=[67.8], lon=[20.3], mode="markers", name="Kiruna"), row=1, col=2
)

fig.update_geos(
    projection_type="stereographic",
    projection_rotation=dict(lat=68, lon=20),
    showland=True,
    landcolor="lightgray",
)
fig.update_geos(
    # Center the map around your points
    # projection_rotation=dict(lat=sum(lats) / len(lats), lon=sum(lons) / len(lons)),
    # lataxis_range=[min(lats) - 1, max(lats) + 1],
    # lonaxis_range=[min(lons) - 1, max(lons) + 1],
    projection_scale=8,  # increase for more zoom
    showland=True,
    landcolor="lightgray",
)

fig.update_layout(template="plotly_white", width=1000, height=1000)
fig.update_xaxes(title_text="EQ Latitude", row=2, col=1)
fig.update_xaxes(title_text="EQ Latitude", row=1, col=1)
fig.update_yaxes(title_text="Grad PV", row=1, col=1)
fig.update_yaxes(title_text="PV", row=2, col=1)
fig.show()

# display(fig)


# --- Callback to link hover ---
def hover_fn(trace, points, state):
    if points.point_inds:
        ind = points.point_inds[0]
        # update hover for other traces
        with fig.batch_update():
            for i, t in enumerate(fig.data):
                # skip the trace that triggered the hover
                if t is trace:
                    continue
                # simulate hover by setting 'selectedpoints' for the matching point
                t.selectedpoints = [ind]


# Connect hover event for all traces
for trace in fig.data:
    trace.on_hover(hover_fn)
