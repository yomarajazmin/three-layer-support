"""Guard against using an in-progress daily bar.

During US market hours yfinance returns today's still-forming bar as the last
row. Its close is a live price and its volume is partial, which would distort
every signal and factor. Until the cutoff (after the 16:00 ET close, with time
for the 16:15 VIX settlement), today's bar is dropped. Daily bar dates are
exchange (New York) dates.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

import config

TZ = "America/New_York"


def now_et() -> pd.Timestamp:
    return pd.Timestamp.now(tz=TZ)


def _cutoff(day: dt.date) -> pd.Timestamp:
    hh, mm = map(int, config.MARKET_DATA_CUTOFF_ET.split(":"))
    return pd.Timestamp(dt.datetime.combine(day, dt.time(hh, mm)), tz=TZ)


def drop_incomplete_bar(df: pd.DataFrame | pd.Series, now: pd.Timestamp | None = None):
    """Drop the last row if it is today's bar and the session hasn't finished yet."""
    now = now or now_et()
    if len(df) and df.index[-1].date() == now.date() and now < _cutoff(now.date()):
        return df.iloc[:-1]
    return df


def _to_et(ts: dt.datetime) -> pd.Timestamp:
    """Naive timestamps (as written by the fetchers) are machine-local time."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize(dt.datetime.now().astimezone().tzinfo)
    return ts.tz_convert(TZ)


def last_completed_session(now: pd.Timestamp | None = None) -> dt.date:
    """Most recent weekday whose daily bar is final (market holidays are not modeled)."""
    now = now or now_et()
    day = now.date() if now >= _cutoff(now.date()) else now.date() - dt.timedelta(days=1)
    while day.weekday() >= 5:
        day -= dt.timedelta(days=1)
    return day


def data_is_stale(as_of: dt.date | pd.Timestamp, downloaded_at: dt.datetime,
                  now: pd.Timestamp | None = None) -> bool:
    """True when data ends before the latest completed session AND was downloaded before that
    session's bar was final. (A download made after the close that still lacks the bar means the
    day was a market holiday, so it is not stale.)"""
    now = now or now_et()
    session = last_completed_session(now)
    return pd.Timestamp(as_of).date() < session and _to_et(downloaded_at) < _cutoff(session)


def download_is_stale(downloaded_at: dt.datetime, now: pd.Timestamp | None = None) -> bool:
    """True if a cached download made before today's cutoff is being reused after it, on a
    weekday: it is missing today's completed bar. (A holiday costs one unneeded re-download.)"""
    now = now or now_et()
    cutoff = _cutoff(now.date())
    return now.weekday() < 5 and _to_et(downloaded_at) < cutoff <= now
