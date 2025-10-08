"""Total point differential feature."""

from __future__ import annotations

import pandas as pd


def compute(ttl_for: pd.Series, ttl_against: pd.Series) -> pd.Series:
    """Return the running difference in total points won."""

    return ttl_for.astype(float) - ttl_against.astype(float)
