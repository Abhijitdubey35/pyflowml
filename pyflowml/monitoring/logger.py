"""
Logger — Colored, structured logging for PyFlowML.
"""

import logging
import sys


COLORS = {
    "DEBUG":    "\033[36m",   # Cyan
    "INFO":     "\033[32m",   # Green
    "WARNING":  "\033[33m",   # Yellow
    "ERROR":    "\033[31m",   # Red
    "CRITICAL": "\033[35m",   # Magenta
    "RESET":    "\033[0m",
}


class ColoredFormatter(logging.Formatter):
    def format(self, record):
        color = COLORS.get(record.levelname, COLORS["RESET"])
        reset = COLORS["RESET"]
        record.msg = f"{color}{record.msg}{reset}"
        return super().format(record)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a named, colored logger.

    Example
    -------
    >>> logger = get_logger("MyModule")
    >>> logger.info("Training started")
    """
    logger = logging.getLogger(f"pyflowml.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ColoredFormatter(
            fmt="%(asctime)s  [%(name)s]  %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
