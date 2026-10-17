"""Signal 4: Credit spreads (proxy).

Spread proxy = log(TLT / HYG): rises when high-yield underperforms Treasuries
(spreads widening). Z-scored against its trailing 1-year mean and std.
Linear map: z = -2 (tight) -> 100, z = +2 (wide) -> 0, clamped.

Caveat: TLT is long duration and HYG is short-to-intermediate, so this proxy
also moves with long-end rates, not only with credit risk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from signals.common import SignalResult, latest, linear_score

KEY = "credit_spreads"
NAME = "Credit Spreads"


def score_history(hyg: pd.Series, tlt: pd.Series, window: int = config.TRADING_DAYS_1Y) -> pd.DataFrame:
    df = pd.concat({"hyg": hyg, "tlt": tlt}, axis=1, sort=True).dropna()
    spread = np.log(df["tlt"] / df["hyg"])
    mean = spread.rolling(window, min_periods=window).mean()
    std = spread.rolling(window, min_periods=window).std()
    z = (spread - mean) / std
    score = linear_score(z, config.CREDIT_Z_AT_100, config.CREDIT_Z_AT_0)
    return pd.DataFrame({"value": z, "spread_proxy": spread, "score": score})


def compute(hyg: pd.Series, tlt: pd.Series) -> SignalResult:
    history = score_history(hyg, tlt)
    row = latest(history)
    state = "tighter than avg" if row["value"] < 0 else "wider than avg"
    return SignalResult(
        key=KEY,
        name=NAME,
        score=float(row["score"]),
        value=float(row["value"]),
        value_label=f"z = {row['value']:+.2f} ({state})",
        as_of=row.name,
        history=history,
        description="z-score of log(TLT/HYG) vs 1y; z −2 → 100, z +2 → 0 (linear).",
    )
