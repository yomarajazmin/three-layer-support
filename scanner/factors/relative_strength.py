"""Factor 3: Relative strength vs SPY.

Spread = stock 20-day return - SPY 20-day return; score = percentile rank of the
spread across the universe.
"""

from __future__ import annotations

import pandas as pd

import config
from scanner.factors import pct_rank

NAME = "Relative Strength"


def compute(close: pd.DataFrame, spy: pd.Series) -> pd.DataFrame:
    n = config.SCAN_RS_DAYS
    stock_ret = close.iloc[-1] / close.shift(n).iloc[-1] - 1
    spy = spy.reindex(close.index).ffill()
    spy_ret = spy.iloc[-1] / spy.shift(n).iloc[-1] - 1
    spread = stock_ret - spy_ret
    return pd.DataFrame({"ret_20d_pct": stock_ret * 100, "rs_spread_pct": spread * 100,
                         "score": pct_rank(spread)})
