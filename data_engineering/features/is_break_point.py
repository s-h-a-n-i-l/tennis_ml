"""Break-point indicator for the perspective player."""

from __future__ import annotations

import pandas as pd


def compute(
    server_is_persp: pd.Series,
    is_game_point_for: pd.Series,
) -> pd.Series:
    """Return whether the upcoming point is a break point."""

    return ((1 - server_is_persp.astype(int)) & is_game_point_for.astype(bool)).astype(int)
