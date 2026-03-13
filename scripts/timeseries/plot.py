import plotly.graph_objects as go
from datetime import datetime


def plot_map(data, period):
    klat = [67.85]
    klon = [20.22]

    labels = ["IRF"]
    fig = go.Figure()

    # Kiruna point

    fig.add_trace(
        go.Scattergeo(
            lon=klon,
            lat=klat,
            text=labels,
            mode="markers+text",
            textposition="top center",
            name="IRF",
            marker=dict(
                size=10,
                color="red",
                symbol="222",
                line=dict(
                    color="black",
                    width=1,
                ),
            ),
        )
    )

    for dt, val in data.items():
        lon = val["lon"]
        lat = val["lat"]
        fig.add_trace(
            go.Scattergeo(
                lon=[lon],
                lat=[lat],
                text=[dt],
                name=datetime.strftime(dt, "%Y-%m-%d %H:%M:%S"),
                marker=dict(size=10, color="black", symbol="4"),
            )
        )

    fig.update_layout(
        title=f"MLS O3 Measurement - Total: {len(data)}",
        geo=dict(
            projection_type="orthographic",
            showland=True,
            landcolor="white",
            showocean=True,
            oceancolor="lightblue",
        ),
    )

    fig.update_geos(
        visible=True,
        resolution=110,
        showcountries=True,
        countrycolor="black",
        showsubunits=True,
        subunitcolor="gray",
        fitbounds="locations",
        showframe=True,
        lataxis=dict(showgrid=True, dtick=3, gridcolor="gray", griddash="dash"),
        lonaxis=dict(showgrid=True, dtick=3, gridcolor="gray", griddash="dash"),
    )
    fig.write_html("map_plot.html")
