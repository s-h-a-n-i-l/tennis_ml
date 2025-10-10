"""Select rolling average serve speed for the perspective player."""

from __future__ import annotations

import pandas as pd

from ._selectors import select


def compute(
    perspective: pd.Series,
    p1_values: pd.Series,
    p2_values: pd.Series,
) -> pd.Series:
    """Return the perspective-specific values."""

    return select(perspective, p1_values, p2_values)
