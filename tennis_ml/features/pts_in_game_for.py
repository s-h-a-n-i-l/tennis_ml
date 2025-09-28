"""Points won in the current game from a perspective."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute(
    perspective: pd.Series,
    p1_points: pd.Series,
    p2_points: pd.Series,
) -> pd.Series:
    """Return perspective specific points won in the game."""

    is_p1 = perspective.astype(str).str.upper().eq("P1")
    return pd.Series(
        np.where(is_p1, p1_points.astype(int), p2_points.astype(int)),
        index=perspective.index,
    )
