import numpy as np
from ozone.io import get_downloadsdir
from ozone.analysis import pressure_thickness, get_period
import matplotlib.pyplot as plt


mira2 = np.load(
    get_downloadsdir() / "MIRA2_screened_matching.npy",
    allow_pickle=True,
).item()

mls = np.load(
    get_downloadsdir() / "MLS_screened_matching.npy",
    allow_pickle=True,
).item()

m2day = get_period(mira2, "day")
mlsday = get_period(mls, "day")
