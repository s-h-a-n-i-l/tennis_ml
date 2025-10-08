"""Ace differential feature."""

from __future__ import annotations

import pandas as pd


def compute(aces_for: pd.Series, aces_against: pd.Series) -> pd.Series:
    """Return the running difference in aces served."""

    return aces_for.astype(float) - aces_against.astype(float)
