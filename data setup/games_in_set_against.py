"""Games won in the current set by the opponent."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute(
    perspective: pd.Series,
    p1_games: pd.Series,
    p2_games: pd.Series,
) -> pd.Series:
    """Return games won in the set by the opponent for each perspective."""

    is_p1 = perspective.astype(str).str.upper().eq("P1")
    return pd.Series(
        np.where(is_p1, p2_games.astype(int), p1_games.astype(int)),
        index=perspective.index,
    )
