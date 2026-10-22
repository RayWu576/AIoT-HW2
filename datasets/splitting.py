"""Reusable holdout and CV strategies; preprocessing receives raw train rows."""
import math
import numpy as np
from sklearn.model_selection import KFold, TimeSeriesSplit, train_test_split


def holdout_indices(dataset, test_size=0.2, *, random_state=0, gap=0):
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if not isinstance(gap, int) or gap < 0:
        raise ValueError("gap must be a nonnegative integer")
    indices = np.arange(len(dataset.y))
    if dataset.config.split_strategy == "chronological":
        dates = dataset.temporal_metadata["date"]
        if dates.isna().any() or not dates.is_monotonic_increasing or dates.duplicated().any():
            raise ValueError("Chronological split requires ordered unique date metadata")
        cutoff = len(indices) - math.ceil(len(indices) * test_size)
        if cutoff <= gap:
            raise ValueError("Not enough training observations for this gap")
        return indices[:cutoff-gap], indices[cutoff:]
    if gap:
        raise ValueError("gap applies only to chronological datasets")
    return train_test_split(indices, test_size=test_size, random_state=random_state)


def make_cv(config, n_splits=5, *, gap=0, random_state=0):
    if config.split_strategy == "chronological":
        return TimeSeriesSplit(n_splits=n_splits, gap=gap)
    if gap:
        raise ValueError("gap applies only to chronological datasets")
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
