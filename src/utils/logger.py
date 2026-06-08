import logging
import os
from datetime import datetime


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger that writes to console and a log file."""
    os.makedirs("logs", exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    log_file = f"logs/{datetime.now().strftime('%Y%m%d')}.log"
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
