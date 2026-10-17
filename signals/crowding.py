"""Signal 6: Factor crowding.

Builds two equal-weight long/short baskets from the S&P 500 universe, rebalanced
at each month-end:
  * Momentum: long top decile / short bottom decile of 12-1 month return.
  * Value (proxy): long-term reversal. Long the bottom decile / short the top
    decile of the 3-year return excluding the most recent year. Price-only, so
    the backtest has no fundamental look-ahead.

Score = linear map of the 63-day rolling correlation between the daily factor
returns: corr +0.3 -> 100 (normal), corr -0.8 -> 0 (extreme crowding).
Highly negative correlation = momentum crowded = reversal risk.
"""

from __future__ import annotations

import pandas as pd

import config
from signals.common import SignalResult, latest, linear_score

KEY = "crowding"
NAME = "Factor Crowding"


def rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last available trading day of each calendar month."""
    s = index.to_series()
    return pd.DatetimeIndex(s.groupby(index.to_period("M")).max().values)


def _baskets(signal: pd.DataFrame, rebal: pd.DatetimeIndex, long_top: bool):
    """Boolean long/short membership on rebalance dates from a ranking signal."""
    sig = signal.loc[rebal]
    ranks = sig.rank(axis=1, pct=True)
    enough = sig.notna().sum(axis=1) >= config.CROWDING_MIN_STOCKS
    top = ranks > 1 - config.CROWDING_DECILE
    bottom = ranks <= config.CROWDING_DECILE
    top.loc[~enough] = False
    bottom.loc[~enough] = False
    return (top, bottom) if long_top else (bottom, top)


def _held(members: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Daily holdings: membership formed at a rebalance close is held from the NEXT day."""
    daily = members.astype(float).reindex(index).ffill().shift(1).fillna(0.0)
    return daily > 0.5


def _leg_return(returns: pd.DataFrame, held: pd.DataFrame) -> pd.Series:
    return returns.where(held).mean(axis=1)


def factor_returns(closes: pd.DataFrame) -> pd.DataFrame:
    """Daily momentum and value L/S factor returns."""
    closes = closes.sort_index()
    returns = closes.pct_change(fill_method=None)
    rebal = rebalance_dates(closes.index)

    mom_signal = closes.shift(config.CROWDING_MOM_SKIP) / closes.shift(config.CROWDING_MOM_LOOKBACK) - 1
    val_signal = closes.shift(config.CROWDING_VALUE_SKIP) / closes.shift(config.CROWDING_VALUE_LOOKBACK) - 1

    out = {}
    for name, signal, long_top in (("momentum", mom_signal, True), ("value", val_signal, False)):
        long_m, short_m = _baskets(signal, rebal, long_top)
        out[name] = (_leg_return(returns, _held(long_m, closes.index))
                     - _leg_return(returns, _held(short_m, closes.index)))
    return pd.DataFrame(out)


def score_history(closes: pd.DataFrame, window: int = config.CROWDING_CORR_WINDOW) -> pd.DataFrame:
    factors = factor_returns(closes).dropna()
    corr = factors["momentum"].rolling(window, min_periods=window).corr(factors["value"])
    score = linear_score(corr, config.CROWDING_CORR_AT_100, config.CROWDING_CORR_AT_0)
    return pd.DataFrame({"value": corr, "momentum_ret": factors["momentum"],
                         "value_ret": factors["value"], "score": score})


def compute(closes: pd.DataFrame) -> SignalResult:
    history = score_history(closes)
    row = latest(history)
    return SignalResult(
        key=KEY,
        name=NAME,
        score=float(row["score"]),
        value=float(row["value"]),
        value_label=f"mom/value corr {row['value']:+.2f} (63d)",
        as_of=row.name,
        history=history,
        description="63d corr of momentum vs value L/S returns; +0.3 → 100, −0.8 → 0 (linear).",
    )
