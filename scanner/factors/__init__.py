"""Scanner factors. Each `compute(...)` returns a DataFrame indexed by ticker with a 0-100 `score` column."""

import pandas as pd


def pct_rank(s: pd.Series) -> pd.Series:
    """Percentile rank (0-100] across the non-null values; ties share the average rank."""
    return s.rank(pct=True) * 100.0
