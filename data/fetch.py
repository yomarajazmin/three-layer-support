"""Market data access via yfinance, plus S&P 500 constituents from Wikipedia.

One daily download (cached on disk for the day) feeds every layer: the macro
gate uses the full history, and the scanner slices its 1-year window from the
same frames, so both always see identical prices.
"""

from __future__ import annotations

import datetime as dt
import pickle
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

import config
from data.market_hours import download_is_stale, drop_incomplete_bar

SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_UA = {"User-Agent": "Mozilla/5.0 (market-deployment-gate; personal research)"}
OHLCV_FIELDS = ("Close", "High", "Volume")


@dataclass
class MarketData:
    macro: pd.DataFrame            # closes for config.MACRO_TICKERS
    sp500: pd.DataFrame            # closes for current S&P 500 members
    sp500_high: pd.DataFrame
    sp500_volume: pd.DataFrame
    constituents: list[str]        # S&P 500 members requested
    downloaded_at: dt.datetime
    missing_tickers: list[str] = field(default_factory=list)   # members with no price data

    def series(self, ticker: str) -> pd.Series:
        if ticker not in self.macro.columns or self.macro[ticker].dropna().empty:
            raise RuntimeError(f"No close data for {ticker}")
        return self.macro[ticker].dropna().rename(ticker)

    def warnings(self) -> list[str]:
        out = []
        missing_macro = [t for t in config.MACRO_TICKERS
                         if t not in self.macro.columns or self.macro[t].dropna().empty]
        if missing_macro:
            out.append(f"No price data for {', '.join(missing_macro)}.")
        n = len(self.constituents)
        if n and len(self.missing_tickers) / n > 1 - config.PRICE_MIN_COVERAGE:
            out.append(f"Price download failed for {len(self.missing_tickers)} of {n} S&P 500 members "
                       f"(e.g. {', '.join(self.missing_tickers[:5])}); breadth, crowding and the scanner "
                       "use the rest.")
        return out


def download_ohlcv(tickers: list[str], fields=OHLCV_FIELDS,
                   period: str = config.HISTORY_PERIOD) -> dict[str, pd.DataFrame]:
    """Daily auto-adjusted bars: {field: DataFrame(date x ticker)}. Tickers with no closes are dropped
    from every field, and today's bar is dropped while the session is still in progress."""
    raw = yf.download(tickers, period=period, interval="1d", auto_adjust=True, progress=False,
                      threads=True, group_by="column")
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no data for {tickers[:5]}...")
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw = drop_incomplete_bar(raw.sort_index())
    out = {}
    for f in fields:
        frame = raw[f]
        if isinstance(frame, pd.Series):  # single ticker on some yfinance versions
            frame = frame.to_frame(tickers[0])
        out[f] = frame
    keep = out["Close"].columns[out["Close"].notna().any()] if "Close" in out else None
    return {f: (df[keep] if keep is not None else df) for f, df in out.items()}


def download_closes(tickers: list[str], period: str = config.HISTORY_PERIOD) -> pd.DataFrame:
    return download_ohlcv(tickers, fields=("Close",), period=period)["Close"]


def get_sp500_tickers() -> list[str]:
    """Current S&P 500 members from Wikipedia, in Yahoo format (BRK.B -> BRK-B)."""
    resp = requests.get(SP500_WIKI_URL, headers=_UA, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text), attrs={"id": "constituents"})
    symbols = tables[0]["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False)
    tickers = sorted(set(symbols))
    if len(tickers) < 400:  # sanity check against a changed page layout
        raise RuntimeError(f"Parsed only {len(tickers)} S&P 500 tickers from Wikipedia")
    return tickers


def _cache_path() -> Path:
    path = Path(config.CACHE_DIR)
    path.mkdir(exist_ok=True)
    return path / f"market_ohlcv_{config.HISTORY_PERIOD}_{dt.date.today().isoformat()}.pkl"


def load_market_data(refresh: bool = False) -> MarketData:
    """All market data, cached on disk for the day. `refresh=True` forces a new download."""
    cache = _cache_path()
    if cache.exists() and not refresh:
        try:
            with cache.open("rb") as fh:
                data = pickle.load(fh)
            if isinstance(data, MarketData) and not download_is_stale(data.downloaded_at):
                return data   # (stale = cached mid-session, now after the close)
        except Exception:      # unreadable / incompatible cache file -> download again
            pass
    tickers = get_sp500_tickers()
    stocks = download_ohlcv(tickers)
    data = MarketData(
        macro=download_closes(config.MACRO_TICKERS),
        sp500=stocks["Close"], sp500_high=stocks["High"], sp500_volume=stocks["Volume"],
        constituents=tickers, downloaded_at=dt.datetime.now(),
        missing_tickers=sorted(set(tickers) - set(stocks["Close"].columns)),
    )
    for old in cache.parent.glob("market_*.pkl"):  # keep only today's file
        old.unlink()
    with cache.open("wb") as fh:
        pickle.dump(data, fh)
    return data
