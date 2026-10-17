"""Factor 1: Momentum crossover.

Qualifies if the 10-day EMA crossed above the 50-day EMA within the last 5
trading days and is still above it today. Qualifying stocks are scored, among
themselves, as the average of percentile(gap) and percentile(3-month return),
where gap = (EMA10 - EMA50) / EMA50. Non-qualifying stocks score 0.
"""

from __future__ import annotations

import pandas as pd

import config
from scanner.factors import pct_rank

NAME = "Momentum Crossover"


def compute(close: pd.DataFrame) -> pd.DataFrame:
    fast = close.ewm(span=config.SCAN_EMA_FAST, adjust=False, min_periods=config.SCAN_EMA_FAST).mean()
    slow = close.ewm(span=config.SCAN_EMA_SLOW, adjust=False, min_periods=config.SCAN_EMA_SLOW).mean()
    above = (fast > slow).astype(float).where(slow.notna() & fast.notna())   # 1 / 0 / NaN
    cross_up = (above == 1) & (above.shift(1) == 0)

    recent = cross_up.tail(config.SCAN_CROSS_LOOKBACK)
    crossed = recent.any() & (above.iloc[-1] == 1)
    # Trading days since the most recent up-cross inside the lookback (0 = today).
    last_pos = recent[::-1].values.argmax(axis=0)
    days_since = pd.Series(last_pos, index=close.columns).where(recent.any())

    gap = (fast.iloc[-1] - slow.iloc[-1]) / slow.iloc[-1]
    ret_3m = close.iloc[-1] / close.shift(config.SCAN_RET_3M_DAYS).iloc[-1] - 1

    score = pd.Series(0.0, index=close.columns)
    if crossed.any():
        c = crossed[crossed].index
        parts = pd.concat([pct_rank(gap[c]), pct_rank(ret_3m[c])], axis=1)
        score[c] = parts.mean(axis=1, skipna=True).fillna(0.0)

    return pd.DataFrame({
        "crossed": crossed,
        "days_since_cross": days_since.where(crossed),
        "ema_gap_pct": gap * 100,
        "ret_3m_pct": ret_3m * 100,
        "score": score,
    })
