"""Factor 4: 52-week high proximity (George & Hwang, 2004).

Ratio = latest close / highest close over the past 252 trading days.
Score: ratio >= 0.95 -> 100, linear down to 0 at ratio 0.70, clamped.
"""

from __future__ import annotations

import pandas as pd

import config
from signals.common import linear_score

NAME = "52-Week High Proximity"


def compute(close: pd.DataFrame) -> pd.DataFrame:
    high_52w = close.tail(config.SCAN_HIGH_DAYS).max()
    ratio = close.iloc[-1] / high_52w
    score = linear_score(ratio, config.SCAN_HIGH_RATIO_AT_100, config.SCAN_HIGH_RATIO_AT_0)
    return pd.DataFrame({"high_52w": high_52w, "high_ratio": ratio, "score": score})
