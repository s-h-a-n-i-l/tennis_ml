"""Sets won by the opponent from the given perspective."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute(
    perspective: pd.Series,
    p1_sets: pd.Series,
    p2_sets: pd.Series,
) -> pd.Series:
    """Return opponent sets before the current set for each perspective."""

    is_p1 = perspective.astype(str).str.upper().eq("P1")
    return pd.Series(
        np.where(is_p1, p2_sets.astype(int), p1_sets.astype(int)),
        index=perspective.index,
    )
