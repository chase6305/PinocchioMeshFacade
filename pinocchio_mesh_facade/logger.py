"""Logging utility module, provides get_logger function for standardized logging output."""
import logging
import sys

LOGGER_NAME = "mesh_extractor"


def get_logger(name: str = None) -> logging.Logger:
    """Get a standardized logger object.

    :param name: Logger name (optional)
    :return: logging.Logger instance
    """
    logger = logging.getLogger(name or LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '[%(asctime)s][%(levelname)s][%(name)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
