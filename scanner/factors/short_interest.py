"""Factor 5: Short interest decline.

Change = shares short (latest report) / shares short (prior month's report) - 1.
Declining short interest = short covering = bullish. Score = percentile rank of
-change across tickers with data; tickers without data get a neutral 50.
"""

from __future__ import annotations

import pandas as pd

import config
from scanner.factors import pct_rank

NAME = "Short Interest Decline"


def compute(short: pd.DataFrame) -> pd.DataFrame:
    prior = short["shares_short_prior"].where(short["shares_short_prior"] > 0)
    change = short["shares_short"] / prior - 1
    score = pct_rank(-change).fillna(config.SCAN_SHORT_MISSING_SCORE)
    return pd.DataFrame({
        "si_change_pct": change * 100,
        "short_pct_float": short["short_pct_float"] * 100,
        "si_date": short["short_date"],
        "si_has_data": change.notna(),
        "score": score,
    })
