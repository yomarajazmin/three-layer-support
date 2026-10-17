"""Signal 2: VIX term structure.

Ratio = VIX / VIX3M. Below 1.0 = contango (calm), above 1.0 = backwardation (stress).
Linear map: 0.85 -> 100, 1.15 -> 0, clamped.
"""

from __future__ import annotations

import pandas as pd

import config
from signals.common import SignalResult, latest, linear_score

KEY = "vix_term_structure"
NAME = "VIX Term Structure"


def score_history(vix: pd.Series, vix3m: pd.Series) -> pd.DataFrame:
    df = pd.concat({"vix": vix, "vix3m": vix3m}, axis=1, sort=True).dropna()
    ratio = df["vix"] / df["vix3m"]
    score = linear_score(ratio, config.TERM_RATIO_AT_100, config.TERM_RATIO_AT_0)
    return pd.DataFrame({"value": ratio, "vix": df["vix"], "vix3m": df["vix3m"], "score": score})


def compute(vix: pd.Series, vix3m: pd.Series) -> SignalResult:
    history = score_history(vix, vix3m)
    row = latest(history)
    regime = "contango" if row["value"] < 1.0 else "backwardation"
    return SignalResult(
        key=KEY,
        name=NAME,
        score=float(row["score"]),
        value=float(row["value"]),
        value_label=f"{row['value']:.3f} ({regime})",
        as_of=row.name,
        history=history,
        description="VIX / VIX3M; 0.85 → 100, 1.15 → 0 (linear).",
    )
