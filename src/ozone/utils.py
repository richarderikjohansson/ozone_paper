from pathlib import Path

def find_downloads():
    home = Path.home()
    downloadsdir = home / "Downloads"
    if not downloadsdir.exists():
        downloadsdir.mkdir()
    return downloadsdir
