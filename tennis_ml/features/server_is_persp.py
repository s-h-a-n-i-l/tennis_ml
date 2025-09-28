"""Server alignment feature for a given perspective."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute(perspective: pd.Series, server_is_p1: pd.Series) -> pd.Series:
    """Return whether the server matches the player's perspective."""

    perspective = perspective.astype(str).str.upper()
    server_is_p1 = server_is_p1.astype(int)
    is_p1_persp = perspective.eq("P1")
    return pd.Series(
        np.where(is_p1_persp, server_is_p1, 1 - server_is_p1), index=perspective.index
    ).astype(int)
