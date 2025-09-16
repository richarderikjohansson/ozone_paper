from .arts import Ycalc
from .io import MIRA2FindAndMake


def cli_commands():

    # will add more
    commands = {"arts": Ycalc,
                "m2make": MIRA2FindAndMake}

    desc = {"arts": "Used for making simulations with pyarts",
            "m2make": "Used to find MIRA2 files from a path and make product files"}

    return commands, desc
