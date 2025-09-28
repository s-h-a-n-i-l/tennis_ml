from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def compute_ttl_diff(
    ttl_p1: pd.Series,
    ttl_p2: pd.Series,
    perspective: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Compute `ttl_diff` (total points won differential) as a pandas Series.

    Parameters
    ----------
    ttl_p1 : pd.Series
        Cumulative points won by Player 1 prior to the current point.
    ttl_p2 : pd.Series
        Cumulative points won by Player 2 prior to the current point.
    perspective : pd.Series, optional
        If provided, returns the differential from each row's perspective:
        - rows where perspective indicates P1 -> ttl_p1 - ttl_p2
        - rows where perspective indicates P2 -> ttl_p2 - ttl_p1
        Accepts strings "P1"/"P2" (case-insensitive), or integers 1/2.

    Returns
    -------
    pd.Series
        Series of ttl_diff aligned to the inputs' index.

    Notes
    -----
    - Inputs are coerced to numeric; non-numeric values become 0.
    - If `perspective` is not provided, returns `ttl_p1 - ttl_p2`.
    """
    p1 = pd.to_numeric(ttl_p1, errors="coerce").fillna(0)
    p2 = pd.to_numeric(ttl_p2, errors="coerce").fillna(0)
    base = p1 - p2

    if perspective is None:
        return base

    pers = perspective
    # Determine which rows are P1 perspective
    if pd.api.types.is_string_dtype(pers) or getattr(pers, "dtype", None) is object:
        s = pers.astype("string").str.upper().str.strip()
        is_p1 = s.eq("P1")
    elif pd.api.types.is_bool_dtype(pers):
        # Interpret True as P1 perspective by convention
        is_p1 = pers.astype(bool)
    else:
        # Numeric or mixed: map 1->P1, 2->P2; unknowns default to P1
        nums = pd.to_numeric(pers, errors="coerce")
        is_p1 = nums.fillna(1).astype(int).eq(1)

    is_p1 = is_p1.reindex(base.index).fillna(True)
    out = pd.Series(
        np.where(is_p1.to_numpy(), base.to_numpy(), (-base).to_numpy()),
        index=base.index,
        dtype=float,
        name="ttl_diff",
    )
    return out


__all__ = ["compute_ttl_diff"]

