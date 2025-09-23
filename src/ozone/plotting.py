import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from .utils import find_downloads
from .io import get_simulationdir, get_screendir
import numpy as np

class Plotting:
    def __init__(self, fig, logger):
        self.logger = logger
        self.ddir = find_downloads

        match fig:
            case "fig01":
                figname = "fig01.pdf"
                self.make_fig01(figname=figname)

    def make_fig01(self, figname):
        filename = get_screendir() / f"{figname}.npy"
        data = np.load(filename, allow_pickle=True).item()



