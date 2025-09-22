import logging


def get_logger() -> logging.Logger:
    """Function to create logger object

    Returns:
        Logger object
    """

    logger = logging.getLogger("runtime_logger")
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_format = logging.Formatter("[%(levelname)s] %(message)s")
    console_handler.setFormatter(console_format)

    logger.addHandler(console_handler)
    logger.propagate = False
    return logger
