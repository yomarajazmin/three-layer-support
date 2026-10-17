"""Signal 3: Market breadth.

% of S&P 500 members trading above their 200-day SMA.
Linear map: 80% -> 100, 30% -> 0, clamped.
Catches narrow rallies driven by a few mega-caps while most stocks decline.

Note: uses *current* index members for the whole history (survivorship bias in
the historical series; the latest reading is unaffected).
"""

from __future__ import annotations

import pandas as pd

import config
from signals.common import SignalResult, latest, linear_score

KEY = "breadth"
NAME = "Market Breadth"


def score_history(closes: pd.DataFrame, sma_days: int = config.BREADTH_SMA_DAYS) -> pd.DataFrame:
    sma = closes.rolling(sma_days, min_periods=sma_days).mean()
    valid = sma.notna() & closes.notna()
    above = (closes > sma) & valid
    n_valid = valid.sum(axis=1)
    pct = (above.sum(axis=1) / n_valid * 100.0).where(n_valid > 0)
    score = linear_score(pct, config.BREADTH_PCT_AT_100, config.BREADTH_PCT_AT_0)
    return pd.DataFrame({"value": pct, "n_stocks": n_valid, "score": score})


def compute(closes: pd.DataFrame) -> SignalResult:
    history = score_history(closes)
    row = latest(history)
    return SignalResult(
        key=KEY,
        name=NAME,
        score=float(row["score"]),
        value=float(row["value"]),
        value_label=f"{row['value']:.1f}% above 200d SMA ({int(row['n_stocks'])} stocks)",
        as_of=row.name,
        history=history,
        description="% of S&P 500 above 200-day SMA; 80% → 100, 30% → 0 (linear).",
    )
