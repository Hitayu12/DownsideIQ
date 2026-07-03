"""Walk-forward validation (spec §18).

Time-ordered, expanding-window splits — NEVER a random split. Train on the past,
test on the immediately following block, then roll forward. This is the only
validation scheme that respects the arrow of time for a trading model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Split:
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray


def walk_forward_splits(
    n: int,
    n_splits: int = 5,
    min_train: int = 252,
    expanding: bool = True,
) -> list[Split]:
    """Generate expanding-window walk-forward splits over ``n`` ordered rows.

    The remaining rows after ``min_train`` are divided into ``n_splits`` equal
    test blocks; each fold trains on everything before its test block.
    """
    if n <= min_train + n_splits:
        # Not enough data for the requested scheme; fall back to a single split.
        cut = max(1, int(n * 0.7))
        return [Split(0, np.arange(cut), np.arange(cut, n))]

    test_total = n - min_train
    block = test_total // n_splits
    splits: list[Split] = []
    for i in range(n_splits):
        test_start = min_train + i * block
        test_end = n if i == n_splits - 1 else min_train + (i + 1) * block
        train_start = 0 if expanding else max(0, test_start - min_train)
        splits.append(Split(
            fold=i,
            train_idx=np.arange(train_start, test_start),
            test_idx=np.arange(test_start, test_end),
        ))
    return splits


def time_train_test_split(index: pd.Index, train_frac: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """Simple time-ordered holdout split (no shuffling)."""
    n = len(index)
    cut = int(n * train_frac)
    return np.arange(cut), np.arange(cut, n)
