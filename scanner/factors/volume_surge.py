"""Factor 2: Volume surge.

Ratio = 5-day average volume / 20-day average volume. Expanding volume suggests
institutional accumulation. Linear map: 0.7 -> 0, 2.0 -> 100, clamped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from signals.common import linear_score

NAME = "Volume Surge"


def compute(volume: pd.DataFrame) -> pd.DataFrame:
    short = volume.tail(config.SCAN_VOL_SHORT).mean()
    long = volume.tail(config.SCAN_VOL_LONG).mean()
    ratio = short / long.replace(0, np.nan)
    score = linear_score(ratio, config.SCAN_VOL_RATIO_AT_100, config.SCAN_VOL_RATIO_AT_0)
    return pd.DataFrame({"vol_ratio": ratio, "score": score})
