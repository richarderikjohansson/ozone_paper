from .arts import Ycalc
from .mira2 import MIRA2FindAndMake
from .analysis import MatchData
from .mls import MLSFindAndMake
from .mls import MLSFindAndMakeTracer
from .screening import DataScreener
from .plotting import Plotting
from types import SimpleNamespace

COLORS = SimpleNamespace(
    {
        "overlay": "#6c7086",
        "mantle": "#181825",
        "red": "#d20f39",
        "peach": "#fe640b",
        "blue": "#1e66f5",
    }
)


def cli_commands():
    # will add more
    commands = {
        "arts": Ycalc,
        "m2make": MIRA2FindAndMake,
        "mlsmake": MLSFindAndMake,
        "screen": DataScreener,
        "plotting": Plotting,
        "match": MatchData,
        "tracersmake": MLSFindAndMakeTracer,
    }

    desc = {
        "arts": "Used for making simulations with pyarts",
        "m2make": "Used to find MIRA2 files and make product files",
        "mlsmake": "Used to find MLS files and make new file with product",
        "screen": "Used to screen data for either MIRA2 or MLS",
        "match": "Used to match MIRA2 and MLS data. This also interpolates MLS data to a coarser grid and convolves the MLS data with the AK from the MIRA2 retrieval",
        "plotting": "Used to plot figures",
        "tracersmake": "Create datasets for tracer-tracer reference function",
    }

    return commands, desc


def figure_methods():
    methods = [("make_fig01", "fig01")]
    return methods
