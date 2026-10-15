"""Reproducibility seed utility for ISLR Phase 0.

Ensures deterministic execution across Python, NumPy, and PyTorch runtimes.
"""

import os
import random
from typing import Optional

DEFAULT_RANDOM_SEED = 42
MIN_VALID_SEED = 0


def set_seed(seed: Optional[int] = None) -> int:
    """Set random seed across all libraries for deterministic execution.

    Args:
        seed: Non-negative integer seed. Defaults to DEFAULT_RANDOM_SEED (42).

    Returns:
        int: The applied random seed.

    Raises:
        ValueError: If seed is negative.
        TypeError: If seed is not an integer.
    """
    effective_seed = DEFAULT_RANDOM_SEED if seed is None else seed
    if type(effective_seed) is bool or not isinstance(effective_seed, int):
        raise TypeError(
            f"Seed must be an integer, got {type(effective_seed).__name__}."
        )
    if effective_seed < MIN_VALID_SEED:
        raise ValueError(
            f"Seed must be greater than or equal to {MIN_VALID_SEED}, got {effective_seed}."
        )

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(effective_seed)

    # Set Python standard random seed
    random.seed(effective_seed)

    # Set NumPy seed if numpy is installed
    try:
        import numpy as np
        np.random.seed(effective_seed)
    except ImportError:
        pass

    # Set PyTorch seed if torch is installed
    try:
        import torch
        torch.manual_seed(effective_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(effective_seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return effective_seed
