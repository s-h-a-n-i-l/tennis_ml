"""Games won in the current set for the perspective player."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute(
    perspective: pd.Series,
    p1_games: pd.Series,
    p2_games: pd.Series,
) -> pd.Series:
    """Return games won in the set for each perspective."""

    is_p1 = perspective.astype(str).str.upper().eq("P1")
    p1 = pd.to_numeric(p1_games, errors="coerce").replace([np.inf, -np.inf], 0).fillna(0).astype(int)
    p2 = pd.to_numeric(p2_games, errors="coerce").replace([np.inf, -np.inf], 0).fillna(0).astype(int)
    return pd.Series(np.where(is_p1, p1, p2), index=perspective.index)
