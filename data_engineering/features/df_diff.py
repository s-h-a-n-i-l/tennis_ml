"""Double-fault differential feature."""

from __future__ import annotations

import pandas as pd


def compute(df_for: pd.Series, df_against: pd.Series) -> pd.Series:
    """Return the running difference in double faults."""

    return df_for.astype(float) - df_against.astype(float)
