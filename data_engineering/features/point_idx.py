"""Feature computation for point sequence index within a match."""

from __future__ import annotations

import pandas as pd


def compute(match_id: pd.Series) -> pd.Series:
    """Return the 1-indexed point number within each match.

    Parameters
    ----------
    match_id:
        Series identifying the match for each row. The Series must already be
        ordered by point chronology within each match.
    """

    return match_id.groupby(match_id, sort=False).cumcount() + 1
