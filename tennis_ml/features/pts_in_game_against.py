"""Points won by the opponent in the current game."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute(
    perspective: pd.Series,
    p1_points: pd.Series,
    p2_points: pd.Series,
) -> pd.Series:
    """Return the opponent's points in the game for each perspective."""

    is_p1 = perspective.astype(str).str.upper().eq("P1")
    return pd.Series(
        np.where(is_p1, p2_points.astype(int), p1_points.astype(int)),
        index=perspective.index,
    )
