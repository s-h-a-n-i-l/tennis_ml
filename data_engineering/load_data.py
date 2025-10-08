from __future__ import annotations

from pathlib import Path
from typing import Union, Iterable

import numpy as np
import pandas as pd


def _read_parquets(files: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for f in files:
        frames.append(pd.read_parquet(f, engine="pyarrow"))
    return pd.concat(frames, ignore_index=True)


def load_training_dataframe(parquet_root: Union[str, Path], label_col: str = "y_match") -> pd.DataFrame:
    """Load and normalize a concatenated training DataFrame from parquet files.

    - Recursively reads ``*.parquet`` under ``parquet_root``
    - Coerces datetimes to seconds (float) and keeps them as floats
    - Replaces inf with NaN, drops rows missing only the label
    - Converts object-like columns to booleans (for 'true'/'false') or category codes
    - Converts numeric columns to the most suitable numeric type
      (booleans for {0,1}, integers when float is integral, otherwise float)
    - Fills remaining NaNs in features with 0 (label is not filled)
    """

    root = Path(parquet_root)
    files = sorted(root.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {root}")

    df = _read_parquets(files)

    # Datetime -> seconds (float)
    dt_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns
    for col in dt_cols:
        s = pd.to_datetime(df[col], errors="coerce")
        df[col] = np.where(s.notna(), s.view("int64") / 1e9, np.nan)

    # Prefer keeping 'time'/'elapsed_time' as float if present
    for col in ("time", "elapsed_time"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    # Normalize text-like columns (except identifiers)
    obj_cols = df.select_dtypes(include=["object", "string", "category"]).columns
    obj_cols = obj_cols.drop([c for c in ("match_id", label_col) if c in obj_cols])
    for col in obj_cols:
        s = df[col].astype("string")
        lower = s.str.lower()
        uniq = set(lower.dropna().unique())
        if uniq and uniq <= {"true", "false"}:
            df[col] = (lower == "true")
        else:
            df[col] = s.astype("category").cat.codes

    # Replace infs; we'll handle NaNs next
    df = df.replace([np.inf, -np.inf], np.nan)

    # Drop rows missing the label only, if present
    if label_col in df.columns:
        df = df.dropna(axis=0, subset=[label_col])

    # Numeric coercion for all number-like cols
    num_cols = df.select_dtypes(include="number").columns
    for col in num_cols:
        if col in ("time", "elapsed_time"):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
            continue

        series = pd.to_numeric(df[col], errors="coerce")
        finite = series.dropna()

        if not finite.empty and pd.api.types.is_float_dtype(series):
            if np.allclose(finite, np.round(finite)):
                series = series.round().astype("Int64")

        finite = series.dropna()
        if not finite.empty:
            values = set(pd.Series(finite).astype(float))
            if values <= {0.0, 1.0}:
                series = series.fillna(0).astype(bool)

        df[col] = series

    # Fill remaining NaNs in features with 0 (don't touch the label)
    fill_cols = [c for c in df.columns if c != label_col]
    df[fill_cols] = df[fill_cols].fillna(0)

    return df.reset_index(drop=True)


__all__ = ["load_training_dataframe"]
