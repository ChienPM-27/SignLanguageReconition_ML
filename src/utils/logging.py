"""Logging utility for ISLR Phase 0 pipeline.

Provides structured logging with configurable log levels and formatting.
"""

import logging
import sys
from typing import Optional

DEFAULT_LOGGER_NAME = "islr"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


def get_logger(name: str = DEFAULT_LOGGER_NAME, level: Optional[str] = None) -> logging.Logger:
    """Return a configured logger with structured formatting.

    Args:
        name: Name of the logger.
        level: Optional log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               Defaults to DEFAULT_LOG_LEVEL if omitted or None.

    Returns:
        logging.Logger: Configured logger instance.
    """
    effective_level_str = (level or DEFAULT_LOG_LEVEL).upper()
    if effective_level_str not in VALID_LOG_LEVELS:
        raise ValueError(
            f"Invalid log level '{effective_level_str}'. "
            f"Must be one of: {sorted(VALID_LOG_LEVELS)}"
        )

    log_level = getattr(logging, effective_level_str)
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Avoid adding duplicate handlers if logger was already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        formatter = logging.Formatter(
            fmt=DEFAULT_LOG_FORMAT,
            datefmt=DEFAULT_DATE_FORMAT,
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        for handler in logger.handlers:
            handler.setLevel(log_level)

    logger.propagate = False
    return logger
