"""Tiebreak indicator feature."""

from __future__ import annotations

import pandas as pd


def compute(is_tiebreak_game: pd.Series) -> pd.Series:
    """Return the binary tiebreak indicator as integers."""

    return is_tiebreak_game.astype(int)
