from .arts import Ycalc
from .mira2 import MIRA2FindAndMake
from .mls import MLSFindAndMake
from .screening import DataScreener


def cli_commands():
    # will add more
    commands = {"arts": Ycalc,
                "m2make": MIRA2FindAndMake,
                "mlsmake": MLSFindAndMake,
                "screen": DataScreener}

    desc = {
        "arts": "Used for making simulations with pyarts",
        "m2make": "Used to find MIRA2 files and make product files",
        "mlsmake": "Used to find MLS files and make new file with product",
        "screen": "Used to screen data for either MIRA2 or MLS"
    }

    return commands, desc
