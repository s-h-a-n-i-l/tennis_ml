"""Sets required to win the match feature."""

from __future__ import annotations

import pandas as pd


def compute(best_of: pd.Series) -> pd.Series:
    """Return the number of sets required to win based on the best-of format."""

    return (best_of.astype(int) // 2) + 1
