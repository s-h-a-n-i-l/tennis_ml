"""Select second-serve in percentage for the opponent."""

from __future__ import annotations

import pandas as pd

from ._selectors import select_opponent


def compute(
    perspective: pd.Series,
    p1_values: pd.Series,
    p2_values: pd.Series,
) -> pd.Series:
    """Return the opponent values for the perspective."""

    return select_opponent(perspective, p1_values, p2_values)
