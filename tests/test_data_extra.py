"""Data layer: constituent parsing, yfinance shapes, cache recovery, MarketData helpers."""

import datetime as dt
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import config
from data import fetch, market_hours

FRI = pd.Timestamp("2026-10-16 20:00", tz=market_hours.TZ)    # after the cutoff: no bar dropping


def wiki_html(symbols):
    rows = "".join(f"<tr><td>{s}</td><td>Name {s}</td></tr>" for s in symbols)
    return ("<html><body><table class='wikitable' id='constituents'><thead><tr><th>Symbol</th><th>Security</th>"
            f"</tr></thead><tbody>{rows}</tbody></table></body></html>")


def fake_get(html, seen):
    def get(url, headers=None, timeout=None):
        seen.update(url=url, headers=headers, timeout=timeout)
        return SimpleNamespace(text=html, raise_for_status=lambda: None)
    return get


def test_sp500_tickers_are_yahoo_formatted_sorted_and_unique(monkeypatch):
    symbols = [f"T{i:03d}" for i in range(450)] + ["BRK.B", "BF.B", "T001"]
    seen = {}
    monkeypatch.setattr(fetch.requests, "get", fake_get(wiki_html(symbols), seen))
    tickers = fetch.get_sp500_tickers()
    assert "BRK-B" in tickers and "BF-B" in tickers and "BRK.B" not in tickers
    assert tickers == sorted(set(tickers)) and len(tickers) == 452
    assert "User-Agent" in seen["headers"] and seen["timeout"]


def test_sp500_tickers_rejects_a_changed_page(monkeypatch):
    monkeypatch.setattr(fetch.requests, "get", fake_get(wiki_html(["AAPL", "MSFT"]), {}))
    with pytest.raises(RuntimeError, match="Parsed only 2"):
        fetch.get_sp500_tickers()


def test_download_ohlcv_single_ticker_series_shape(monkeypatch):
    idx = pd.to_datetime(["2026-10-14", "2026-10-15"])
    raw = pd.DataFrame({"Close": [1.0, 2.0], "High": [1.1, 2.1], "Volume": [10.0, 20.0]}, index=idx)
    monkeypatch.setattr(fetch.yf, "download", lambda *a, **k: raw)
    monkeypatch.setattr(market_hours, "now_et", lambda: FRI)
    out = fetch.download_ohlcv(["ONLY"])
    assert list(out["Close"].columns) == ["ONLY"] and out["Volume"]["ONLY"].tolist() == [10.0, 20.0]


def test_download_ohlcv_drops_tickers_without_closes_from_every_field(monkeypatch):
    idx = pd.to_datetime(["2026-10-14", "2026-10-15"])
    close = pd.DataFrame({"A": [1.0, 2.0], "DEAD": [np.nan, np.nan]}, index=idx)
    raw = pd.concat({"Close": close, "High": close.fillna(5.0), "Volume": close.fillna(7.0)}, axis=1)
    monkeypatch.setattr(fetch.yf, "download", lambda *a, **k: raw)
    monkeypatch.setattr(market_hours, "now_et", lambda: FRI)
    out = fetch.download_ohlcv(["A", "DEAD"])
    assert all(list(df.columns) == ["A"] for df in out.values())


def test_download_ohlcv_raises_on_empty(monkeypatch):
    monkeypatch.setattr(fetch.yf, "download", lambda *a, **k: pd.DataFrame())
    with pytest.raises(RuntimeError, match="no data"):
        fetch.download_ohlcv(["A"])


def test_corrupt_market_cache_triggers_download(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CACHE_DIR", str(tmp_path))
    fetch._cache_path().write_bytes(b"not a pickle")
    monkeypatch.setattr(fetch, "get_sp500_tickers", lambda: ["A", "B"])
    monkeypatch.setattr(fetch, "download_ohlcv", lambda t: {f: pd.DataFrame({"A": [1.0]}) for f in fetch.OHLCV_FIELDS})
    monkeypatch.setattr(fetch, "download_closes", lambda t: pd.DataFrame({config.SPY: [1.0]}))
    m = fetch.load_market_data()
    assert m.constituents == ["A", "B"] and m.missing_tickers == ["B"]
    assert isinstance(fetch.load_market_data(), fetch.MarketData)   # the rewritten cache now loads


def test_market_data_series():
    idx = pd.bdate_range(end="2026-10-16", periods=3)
    m = fetch.MarketData(pd.DataFrame({config.SPY: [1.0, np.nan, 3.0], config.VIX: np.nan}, index=idx),
                         pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [], dt.datetime.now())
    assert m.series(config.SPY).tolist() == [1.0, 3.0] and m.series(config.SPY).name == config.SPY
    with pytest.raises(RuntimeError):
        m.series(config.VIX)          # all NaN
    with pytest.raises(RuntimeError):
        m.series("NOPE")


def test_to_et_handles_naive_and_aware():
    aware = pd.Timestamp("2026-10-16 12:00", tz="UTC")
    assert market_hours._to_et(aware) == aware
    naive = dt.datetime(2026, 10, 16, 12, 0)
    assert market_hours._to_et(naive).tzinfo is not None
