"""Tests for logging utility (R4).

Covers logger instantiation, level configuration, handler reuse, and invalid level handling.
"""

import logging
import pytest

from src.utils.logging import (
    DEFAULT_LOGGER_NAME,
    VALID_LOG_LEVELS,
    get_logger,
)


def test_get_logger_default() -> None:
    """Verify get_logger returns logger with DEFAULT_LOGGER_NAME and INFO level."""
    logger = get_logger()
    assert logger.name == DEFAULT_LOGGER_NAME
    assert logger.level == logging.INFO
    assert len(logger.handlers) >= 1
    assert logger.propagate is False
    assert "INFO" in VALID_LOG_LEVELS


def test_get_logger_custom_name_and_level() -> None:
    """Verify get_logger with custom name and DEBUG level."""
    logger = get_logger(name="custom_test_logger", level="DEBUG")
    assert logger.name == "custom_test_logger"
    assert logger.level == logging.DEBUG


def test_get_logger_case_insensitive_level() -> None:
    """Verify log level name is case-insensitive (e.g. 'warning')."""
    logger = get_logger(name="case_test_logger", level="warning")
    assert logger.level == logging.WARNING


def test_get_logger_invalid_level_raises_value_error() -> None:
    """Verify invalid log level name raises ValueError."""
    with pytest.raises(ValueError, match="Invalid log level 'VERBOSE'"):
        get_logger(name="invalid_level_logger", level="VERBOSE")


def test_get_logger_handler_reuse() -> None:
    """Verify multiple calls to get_logger for same name reuse existing handler without duplication."""
    logger1 = get_logger(name="reuse_logger", level="INFO")
    handler_count_1 = len(logger1.handlers)
    logger2 = get_logger(name="reuse_logger", level="DEBUG")
    handler_count_2 = len(logger2.handlers)

    assert logger1 is logger2
    assert handler_count_1 == handler_count_2
    assert logger2.level == logging.DEBUG
