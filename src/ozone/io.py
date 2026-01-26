from pathlib import Path
from numpy.typing import NDArray
import numpy as np


def get_localdir():
    home = Path.home()
    local = home / ".local"
    ozonepaper = local / "ozonepaper"

    if not ozonepaper.exists():
        ozonepaper.mkdir()

    return ozonepaper


def get_downloadsdir():
    home = Path.home()
    ddir = home / "Downloads"
    return ddir


def get_datadir() -> Path:
    """Function to get path of data directory

    Returns:
        Path to data directory
    """
    file = Path(__file__)
    src = file.parent
    datadir = src / "data"

    return datadir


def get_exportdir() -> Path:
    """Function that returns export directory

    This function will create the directory "m2export"
    within the systems .cache directory. If it does
    not exist and will return this path

    Returns:
       Absolute path to the export directory
    """
    home = Path.home()
    cache = home / ".cache"
    m2export = cache / "m2export"

    if not m2export.exists():
        m2export.mkdir()

    return m2export


def get_screendir() -> Path:
    """Function that returns screen directory

    This function will create the directory "screen"
    within the systems .cache directory. If it does
    not exist and will return this path

    Returns:
       Absolute path to the export directory
    """
    home = Path.home()
    cache = home / ".cache"
    screen = cache / "screen"

    if not screen.exists():
        screen.mkdir()

    return screen


def get_simulationdir() -> Path:
    """Function that returns export directory

    This function will create the directory "simulation"
    within the systems .cache directory. If it does
    not exist and will return this path

    Returns:
       Absolute path to the export directory
    """

    home = Path.home()
    cache = home / ".cache"
    simulation = cache / "simulation"

    if not simulation.exists():
        simulation.mkdir()

    return simulation


def get_data_files_root(ext: str):
    cwd = Path(__file__)
    parents = cwd.parents
    for p in parents:
        root = p / ".git"
        if root.exists():
            datadir = root.parent / "data"

    assert datadir.exists()
    files = datadir.rglob(pattern=f"*.{ext}")
    return [file for file in files], datadir


def get_egdefiles(root: Path) -> NDArray:
    """Find polarvortex edge files from root,

    This function will recursivly find '.dat' files from root directory

    :param root: starting directory
    :return:
    """
    root = Path(root)
    assert root.exists() and root.is_dir()

    it = root.rglob("*.dat")
    files = np.array([file for file in it])
    return files
