"""Best-of match length feature."""

from __future__ import annotations

import pandas as pd


def compute(best_of_value: int, index: pd.Index) -> pd.Series:
    """Return a constant best-of series aligned with the provided index."""

    return pd.Series(best_of_value, index=index, dtype=int)
