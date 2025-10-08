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
    p1 = pd.to_numeric(p1_points, errors="coerce").replace([np.inf, -np.inf], 0).fillna(0).astype(int)
    p2 = pd.to_numeric(p2_points, errors="coerce").replace([np.inf, -np.inf], 0).fillna(0).astype(int)
    return pd.Series(np.where(is_p1, p2, p1), index=perspective.index)
