"""Historical backtest of the deployment gate against SPY.

The composite score is recomputed for every day of history. To avoid
look-ahead, the allocation for day t uses the score from the close of
day t-1: SPY's return from close(t-1) to close(t) is attributed to the zone
implied by score(t-1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config
from signals.composite import zone_for


@dataclass
class BacktestResult:
    daily: pd.DataFrame       # per day: spy, score_prev, zone, sizing, spy_ret, strategy_ret
    by_zone: pd.DataFrame     # per zone stats on SPY returns
    summary: pd.DataFrame     # SPY buy & hold vs zone-sized strategy


def _max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns).cumprod()
    return float((equity / equity.cummax() - 1).min())


def run(composite_score: pd.Series, spy: pd.Series, days: int = config.BACKTEST_DAYS) -> BacktestResult:
    df = pd.concat({"score": composite_score, "spy": spy}, axis=1, sort=True).dropna()
    df["spy_ret"] = df["spy"].pct_change(fill_method=None)
    df["score_prev"] = df["score"].shift(1)          # yesterday's score -> today's allocation
    df = df.dropna(subset=["spy_ret", "score_prev"]).tail(days)
    if df.empty:
        raise ValueError("No overlapping history between composite score and SPY")

    zones = df["score_prev"].map(zone_for)
    df["zone"] = zones.map(lambda z: z["name"])
    df["sizing"] = zones.map(lambda z: z["sizing"])
    df["strategy_ret"] = df["sizing"] * df["spy_ret"]   # undeployed capital earns 0 (no cash yield)

    rows = []
    for zone in config.ZONES:
        r = df.loc[df["zone"] == zone["name"], "spy_ret"]
        rows.append({
            "Zone": zone["name"],
            "Days": len(r),
            "% of days": len(r) / len(df) * 100,
            "Avg daily SPY return %": r.mean() * 100 if len(r) else np.nan,
            "Annualized SPY return %": r.mean() * config.TRADING_DAYS_1Y * 100 if len(r) else np.nan,
            "Hit rate %": (r > 0).mean() * 100 if len(r) else np.nan,
            "Annualized vol %": r.std() * np.sqrt(config.TRADING_DAYS_1Y) * 100 if len(r) > 1 else np.nan,
        })
    by_zone = pd.DataFrame(rows).set_index("Zone")

    summary = pd.DataFrame({
        name: {
            "Total return %": ((1 + r).prod() - 1) * 100,
            "Annualized vol %": r.std() * np.sqrt(config.TRADING_DAYS_1Y) * 100,
            "Max drawdown %": _max_drawdown(r) * 100,
        }
        for name, r in (("SPY buy & hold", df["spy_ret"]), ("Zone-sized strategy", df["strategy_ret"]))
    }).T

    return BacktestResult(daily=df, by_zone=by_zone, summary=summary)
