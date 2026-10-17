"""Shared helpers and the result type every signal returns."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SignalResult:
    key: str              # matches config.SIGNAL_WEIGHTS
    name: str             # display name
    score: float          # latest score, 0-100
    value: float          # latest raw input value
    value_label: str      # formatted raw value for display
    as_of: pd.Timestamp   # date of the latest observation
    history: pd.DataFrame  # columns: "value", "score" (plus any extras), indexed by date
    description: str      # one-line explanation of the mapping


def linear_score(x, at_100: float, at_0: float):
    """Linearly map x so that x == at_100 -> 100 and x == at_0 -> 0, clamped to [0, 100].

    Works in either direction (at_100 may be above or below at_0) and on
    scalars, numpy arrays or pandas objects.
    """
    scaled = (x - at_0) / (at_100 - at_0) * 100.0
    return np.clip(scaled, 0.0, 100.0)


def latest(history: pd.DataFrame) -> pd.Series:
    """Last row with a valid score."""
    valid = history.dropna(subset=["score"])
    if valid.empty:
        raise ValueError("Not enough history to compute a score")
    return valid.iloc[-1]
