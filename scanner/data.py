"""Scanner data: the last year of daily OHLCV for the S&P 500 + SPY, and short interest snapshots.

Prices are sliced from the shared market download (data.fetch.load_market_data),
so the scanner and the macro gate always use the same bars.
"""

from __future__ import annotations

import datetime as dt
import pickle
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yfinance as yf

import config
from data.fetch import MarketData, load_market_data


@dataclass
class ScanData:
    close: pd.DataFrame     # date x ticker
    high: pd.DataFrame
    volume: pd.DataFrame
    spy_close: pd.Series
    downloaded_at: dt.datetime
    warnings: list[str] = field(default_factory=list)


def _cache(name: str) -> Path:
    path = Path(config.CACHE_DIR)
    path.mkdir(exist_ok=True)
    return path / f"{name}_{dt.date.today().isoformat()}.pkl"


def _cached(name: str, refresh: bool, build, is_stale=lambda obj: False):
    """Load today's pickle for `name`, or build + save it (dropping older days' files).
    A cached object for which `is_stale(obj)` is true is rebuilt."""
    path = _cache(name)
    if path.exists() and not refresh:
        with path.open("rb") as fh:
            obj = pickle.load(fh)
        if not is_stale(obj):
            return obj
    obj = build()
    for old in path.parent.glob(f"{name}_*.pkl"):
        old.unlink()
    with path.open("wb") as fh:
        pickle.dump(obj, fh)
    return obj


def from_market(m: MarketData) -> ScanData:
    start = m.sp500.index[-1] - pd.Timedelta(days=config.SCANNER_LOOKBACK_DAYS)
    close = m.sp500.loc[start:]
    keep = close.columns[close.notna().any()]
    return ScanData(close[keep], m.sp500_high.loc[start:, keep], m.sp500_volume.loc[start:, keep],
                    m.series(config.SPY).loc[start:], m.downloaded_at, m.warnings())


def load_ohlcv(refresh: bool = False) -> ScanData:
    return from_market(load_market_data(refresh=refresh))


def _short_interest_one(ticker: str) -> dict:
    error = None
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception as exc:   # rate limits / network: recorded so the scan can warn about them
        info, error = {}, type(exc).__name__

    def date(key):
        ts = info.get(key)
        return pd.Timestamp(ts, unit="s").normalize() if ts else pd.NaT

    return {
        "ticker": ticker,
        "shares_short": info.get("sharesShort"),
        "shares_short_prior": info.get("sharesShortPriorMonth"),
        "short_date": date("dateShortInterest"),
        "short_prior_date": date("sharesShortPreviousMonthDate"),
        "short_pct_float": info.get("shortPercentOfFloat"),
        "si_error": error,
    }


def load_short_interest(tickers: list[str], refresh: bool = False) -> pd.DataFrame:
    """Latest vs prior short interest per ticker (one yfinance .info call each, cached daily)."""
    def build() -> pd.DataFrame:
        with ThreadPoolExecutor(max_workers=config.SCAN_SHORT_WORKERS) as pool:
            rows = list(pool.map(_short_interest_one, tickers))
        df = pd.DataFrame(rows).set_index("ticker")
        for col in ("shares_short", "shares_short_prior", "short_pct_float"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    # A cache where most lookups errored (e.g. rate-limited) is retried rather than reused all day.
    many_errors = lambda df: ("si_error" not in df or
                              df["si_error"].notna().mean() > 1 - config.SCAN_SHORT_MIN_COVERAGE)
    df = _cached("short_interest", refresh, build, is_stale=many_errors)
    missing = [t for t in tickers if t not in df.index]
    if missing:  # universe changed since the cache was written
        df = pd.concat([df, pd.DataFrame([_short_interest_one(t) for t in missing]).set_index("ticker")])
    return df.loc[tickers]


def short_interest_warning(short: pd.DataFrame, has_data: pd.Series) -> str | None:
    """Warning text when too few tickers have usable short interest (they score a neutral 50)."""
    n, covered = len(has_data), int(has_data.sum())
    if not n or covered / n >= config.SCAN_SHORT_MIN_COVERAGE:
        return None
    errors = int(short["si_error"].notna().sum()) if "si_error" in short else 0
    return (f"Short interest available for only {covered} of {n} stocks ({errors} request errors, "
            f"e.g. rate limiting); the rest score a neutral {config.SCAN_SHORT_MISSING_SCORE:.0f}.")
