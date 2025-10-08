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
    p1 = pd.to_numeric(p1_points, errors="coerce").replace([np.inf, -np.inf], 0).fillna(0).astype(int)
    p2 = pd.to_numeric(p2_points, errors="coerce").replace([np.inf, -np.inf], 0).fillna(0).astype(int)
    return pd.Series(np.where(is_p1, p1, p2), index=perspective.index)
