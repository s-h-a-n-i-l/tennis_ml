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
    p1 = pd.to_numeric(p1_sets, errors="coerce").replace([np.inf, -np.inf], 0).fillna(0).astype(int)
    p2 = pd.to_numeric(p2_sets, errors="coerce").replace([np.inf, -np.inf], 0).fillna(0).astype(int)
    return pd.Series(np.where(is_p1, p2, p1), index=perspective.index)
