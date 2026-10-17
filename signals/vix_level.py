"""Signal 1: VIX level.

Percentile-rank of today's VIX against the trailing 1-year window (current day
included). Low VIX = high score: score = 100 - percentile.
Bonus +5 if VIX < 15, penalty -10 if VIX > 30, then clamped to [0, 100].
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from signals.common import SignalResult, latest

KEY = "vix_level"
NAME = "VIX Level"


def score_history(vix: pd.Series, window: int = config.TRADING_DAYS_1Y) -> pd.DataFrame:
    vix = vix.dropna()
    # Fraction of the window's values <= today's value (ties share the max rank).
    percentile = vix.rolling(window, min_periods=window).rank(pct=True, method="max") * 100.0
    score = 100.0 - percentile
    score = score + np.where(vix < config.VIX_LOW_BONUS_LEVEL, config.VIX_LOW_BONUS, 0.0)
    score = score + np.where(vix > config.VIX_HIGH_PENALTY_LEVEL, config.VIX_HIGH_PENALTY, 0.0)
    score = score.clip(0.0, 100.0).where(percentile.notna())
    return pd.DataFrame({"value": vix, "percentile": percentile, "score": score})


def _ordinal(x: float) -> str:
    n = int(round(x))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def compute(vix: pd.Series) -> SignalResult:
    history = score_history(vix)
    row = latest(history)
    return SignalResult(
        key=KEY,
        name=NAME,
        score=float(row["score"]),
        value=float(row["value"]),
        value_label=f"{row['value']:.2f} ({_ordinal(row['percentile'])} pctile, 1y)",
        as_of=row.name,
        history=history,
        description="100 − 1y percentile of VIX; +5 if VIX < 15, −10 if VIX > 30.",
    )
