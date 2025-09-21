from pathlib import Path


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

    This function will create the directory "export"
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








