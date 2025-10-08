"""Game-point indicator for the opponent."""

from __future__ import annotations

import pandas as pd

from .is_game_point_for import compute as compute_for


def compute(
    pts_for: pd.Series,
    pts_against: pd.Series,
    is_tiebreak: pd.Series,
) -> pd.Series:
    """Return the opponent's game-point indicator."""

    return compute_for(pts_against, pts_for, is_tiebreak)
