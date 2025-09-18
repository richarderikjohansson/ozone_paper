from pathlib import Path
import configparser
from .logger import get_logger


class DataScreener:
    def __init__(self, source, screen_file):
        self.source = Path(source).resolve()
        self.sf = Path(screen_file).resolve()
        self.logger = get_logger()


