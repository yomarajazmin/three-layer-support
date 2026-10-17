"""Signal 5: Put/Call sentiment (proxy).

Uses VIX 20-day rate of change as a fear proxy. Rapidly rising VIX = fear = low score.
Linear map: ROC -30% -> 100, ROC +50% -> 0, clamped.
"""

from __future__ import annotations

import pandas as pd

import config
from signals.common import SignalResult, latest, linear_score

KEY = "put_call"
NAME = "Put/Call Sentiment"


def score_history(vix: pd.Series, days: int = config.PUTCALL_ROC_DAYS) -> pd.DataFrame:
    vix = vix.dropna()
    roc = vix.pct_change(days) * 100.0
    score = linear_score(roc, config.PUTCALL_ROC_AT_100, config.PUTCALL_ROC_AT_0)
    return pd.DataFrame({"value": roc, "score": score})


def compute(vix: pd.Series) -> SignalResult:
    history = score_history(vix)
    row = latest(history)
    return SignalResult(
        key=KEY,
        name=NAME,
        score=float(row["score"]),
        value=float(row["value"]),
        value_label=f"VIX 20d ROC {row['value']:+.1f}%",
        as_of=row.name,
        history=history,
        description="VIX 20-day % change; −30% → 100, +50% → 0 (linear).",
    )
