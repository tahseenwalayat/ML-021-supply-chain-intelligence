import os
import sys
import logging
from typing import Optional

# Load environment variables from .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str = "supply_chain",
    level: Optional[str] = None,
    log_format: str = DEFAULT_LOG_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT
) -> logging.Logger:
    """
    Returns a configured logger with standard stream handler and formatting.

    Usage:
        from src.utils.logging_config import get_logger
        logger = get_logger("my_module")
    """
    logger = logging.getLogger(name)

    # Determine log level priority: explicit arg -> ENV var -> INFO
    if level is None:
        level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    else:
        level_name = str(level).upper()

    log_level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(log_level)

    # Avoid adding duplicate handlers if logger is already configured
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Prevent double logging propagation to root logger if handlers exist
    logger.propagate = False

    return logger
