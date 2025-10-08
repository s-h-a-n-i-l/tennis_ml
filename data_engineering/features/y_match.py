"""Match-level outcome target feature."""

from __future__ import annotations

import pandas as pd


def compute(perspective: pd.Series, match_winner: pd.Series) -> pd.Series:
    """Return binary outcome flag for the given perspective."""

    perspective = perspective.astype(str).str.upper()
    match_winner = match_winner.astype(float)
    win_as_p1 = perspective.eq("P1") & (match_winner == 1)
    win_as_p2 = perspective.eq("P2") & (match_winner == 2)
    return (win_as_p1 | win_as_p2).astype(int)
