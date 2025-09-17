from pathlib import Path


def exportdir() -> Path:
    """Function that returns export directory

    This function will create the directory "export"
    within the code-base if it does not exist and will
    return this path

    Returns:
       Absolute path to the export directory
    """
    filedir = Path(__file__).resolve()
    ddir = filedir.parent / "data"
    edir = ddir / "export"
    if not edir.exists():
        edir.mkdir()

    return edir
