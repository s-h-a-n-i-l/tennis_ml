"""Helpers for selecting player-specific values based on perspective."""

from __future__ import annotations

import numpy as np
import pandas as pd


def select(
    perspective: pd.Series,
    when_p1: pd.Series | np.ndarray,
    when_p2: pd.Series | np.ndarray,
) -> pd.Series:
    """Return a series picking values for the current perspective."""

    mask = perspective.astype(str).to_numpy() == "P1"
    p1_values = np.asarray(when_p1)
    p2_values = np.asarray(when_p2)
    return pd.Series(np.where(mask, p1_values, p2_values), index=perspective.index)


def select_opponent(
    perspective: pd.Series,
    when_p1: pd.Series | np.ndarray,
    when_p2: pd.Series | np.ndarray,
) -> pd.Series:
    """Return opponent values for the current perspective."""

    return select(perspective, when_p2, when_p1)
