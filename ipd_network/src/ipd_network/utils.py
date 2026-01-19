"""Utility helpers for logging and reproducibility."""

import logging


def get_logger(name, level=logging.INFO):
    """Return a configured logger with a basic format."""
    logger = logging.getLogger(name)
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        )
    logger.setLevel(level)
    return logger


def set_random_seed(seed):
    """Set the random seed for standard library and numpy if available."""
    if seed is None:
        return

    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        # numpy is optional for this skeleton
        pass
