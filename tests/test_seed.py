"""Tests for reproducibility seed utility (R4).

Covers deterministic seeding, boundary conditions, negative seed rejection, and type safety.
"""

import os
import random
import pytest

from src.utils.seed import (
    DEFAULT_RANDOM_SEED,
    MIN_VALID_SEED,
    set_seed,
)


def test_set_seed_default() -> None:
    """Verify set_seed with None applies DEFAULT_RANDOM_SEED (42)."""
    applied = set_seed(None)
    assert applied == DEFAULT_RANDOM_SEED
    assert os.environ["PYTHONHASHSEED"] == str(DEFAULT_RANDOM_SEED)


def test_set_seed_custom_value() -> None:
    """Verify set_seed applies custom seed deterministically."""
    custom_seed = 12345
    applied = set_seed(custom_seed)
    assert applied == custom_seed
    assert os.environ["PYTHONHASHSEED"] == str(custom_seed)

    # Test random determinism
    r1 = [random.random() for _ in range(5)]
    set_seed(custom_seed)
    r2 = [random.random() for _ in range(5)]
    assert r1 == r2


def test_set_seed_zero_boundary() -> None:
    """Verify seed=0 is valid non-negative lower bound."""
    applied = set_seed(MIN_VALID_SEED)
    assert applied == MIN_VALID_SEED


def test_set_seed_negative_raises_value_error() -> None:
    """Verify negative seed raises ValueError."""
    with pytest.raises(ValueError, match="Seed must be greater than or equal to 0"):
        set_seed(-1)


def test_set_seed_boolean_type_rejection() -> None:
    """Verify boolean seed raises TypeError instead of being coerced to 0/1."""
    with pytest.raises(TypeError, match="Seed must be an integer, got bool"):
        set_seed(True)  # type: ignore[arg-type]


def test_set_seed_string_type_rejection() -> None:
    """Verify string seed raises TypeError."""
    with pytest.raises(TypeError, match="Seed must be an integer, got str"):
        set_seed("42")  # type: ignore[arg-type]
