"""Game-point indicator for the perspective player."""

from __future__ import annotations

import pandas as pd


def compute(
    pts_for: pd.Series,
    pts_against: pd.Series,
    is_tiebreak: pd.Series,
) -> pd.Series:
    """Return a binary indicator for whether the upcoming point is a game point."""

    is_tb = is_tiebreak.astype(bool)
    cond_regular = (pts_for == 3) & (pts_against <= 2) & (~is_tb)
    cond_deuce = (pts_for >= 3) & (pts_against >= 3) & (pts_for == pts_against + 1) & (~is_tb)
    return (cond_regular | cond_deuce).astype(int)
