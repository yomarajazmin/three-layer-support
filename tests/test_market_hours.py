import datetime as dt
import pickle

import pandas as pd
import pytest

import config
from data import fetch, market_hours
from data.market_hours import TZ, download_is_stale, drop_incomplete_bar
from scanner import data as scan_data

FRI = "2026-10-16"   # a trading day (Friday)


def et(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=TZ)


def bars(last_day: str, n: int = 3) -> pd.DataFrame:
    return pd.DataFrame({"X": range(n)}, index=pd.bdate_range(end=last_day, periods=n))


@pytest.mark.parametrize("now, dropped", [
    (f"{FRI} 09:45", True),    # session in progress
    (f"{FRI} 16:05", True),    # closed, but before the VIX settle / cutoff
    (f"{FRI} 16:30", False),   # at cutoff: bar is final
    (f"{FRI} 20:00", False),
])
def test_todays_bar_dropped_until_cutoff(now, dropped):
    df = bars(FRI)
    out = drop_incomplete_bar(df, now=et(now))
    assert len(out) == (2 if dropped else 3)
    if dropped:
        assert out.index[-1] < pd.Timestamp(FRI)


def test_previous_days_bar_is_kept():
    df = bars("2026-10-15")
    assert len(drop_incomplete_bar(df, now=et(f"{FRI} 10:00"))) == 3        # before today's open
    assert len(drop_incomplete_bar(df, now=et("2026-10-17 11:00"))) == 3     # weekend


def test_series_and_empty_inputs():
    s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-10-15", FRI]))
    assert list(drop_incomplete_bar(s, now=et(f"{FRI} 12:00"))) == [1.0]
    assert drop_incomplete_bar(pd.DataFrame(), now=et(f"{FRI} 12:00")).empty


def test_cutoff_is_configurable(monkeypatch):
    monkeypatch.setattr(config, "MARKET_DATA_CUTOFF_ET", "16:00")
    assert len(drop_incomplete_bar(bars(FRI), now=et(f"{FRI} 16:05"))) == 3


@pytest.mark.parametrize("downloaded, now, stale", [
    (f"{FRI} 10:00", f"{FRI} 17:00", True),     # cached mid-session, reused after the close
    (f"{FRI} 10:00", f"{FRI} 14:00", False),    # still mid-session: nothing newer to get
    (f"{FRI} 16:45", f"{FRI} 20:00", False),    # cached after the close
    ("2026-10-17 09:00", "2026-10-17 18:00", False),   # Saturday: no session
])
def test_download_staleness(downloaded, now, stale):
    assert download_is_stale(et(downloaded), now=et(now)) is stale


def test_naive_download_time_is_treated_as_local():
    local_now = dt.datetime.now()
    assert download_is_stale(local_now, now=et(local_now.astimezone().isoformat())) is False


# ---------------------------------------------------------------- wired into the fetchers
def test_download_closes_drops_partial_bar(monkeypatch):
    idx = pd.to_datetime(["2026-10-15", FRI])
    raw = pd.concat({"Close": pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]}, index=idx)}, axis=1)
    monkeypatch.setattr(fetch.yf, "download", lambda *a, **k: raw)
    monkeypatch.setattr(market_hours, "now_et", lambda: et(f"{FRI} 11:00"))
    out = fetch.download_closes(["A", "B"])
    assert list(out.index) == [pd.Timestamp("2026-10-15")]


def test_download_ohlcv_drops_partial_bar_in_every_field(monkeypatch):
    idx = pd.to_datetime(["2026-10-15", FRI])
    frame = pd.DataFrame({"A": [1.0, 2.0], "B": [5.0, 6.0]}, index=idx)
    raw = pd.concat({"Close": frame, "High": frame, "Volume": frame * 100}, axis=1)
    monkeypatch.setattr(fetch.yf, "download", lambda *a, **k: raw)
    monkeypatch.setattr(market_hours, "now_et", lambda: et(f"{FRI} 11:00"))
    out = fetch.download_ohlcv(["A", "B"])
    assert all(df.index[-1] == pd.Timestamp("2026-10-15") for df in out.values())


def test_market_cache_redownloads_when_stale(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CACHE_DIR", str(tmp_path))
    empty = pd.DataFrame()
    old = fetch.MarketData(pd.DataFrame({"X": [1.0]}), empty, empty, empty, ["A"],
                           downloaded_at=dt.datetime(2026, 10, 16, 10))
    with fetch._cache_path().open("wb") as fh:
        pickle.dump(old, fh)
    calls = []
    monkeypatch.setattr(fetch, "download_closes", lambda t: calls.append(t) or pd.DataFrame({"X": [2.0]}))
    monkeypatch.setattr(fetch, "download_ohlcv", lambda t: calls.append(t) or
                        {f: pd.DataFrame({"A": [1.0]}) for f in fetch.OHLCV_FIELDS})
    monkeypatch.setattr(fetch, "get_sp500_tickers", lambda: ["A"])

    monkeypatch.setattr(fetch, "download_is_stale", lambda d: False)
    assert fetch.load_market_data().macro["X"].iloc[0] == 1.0 and not calls      # fresh cache reused
    monkeypatch.setattr(fetch, "download_is_stale", lambda d: True)
    assert fetch.load_market_data().macro["X"].iloc[0] == 2.0 and len(calls) == 2   # stocks + macro  # stale -> re-download


def test_scanner_cache_rebuilds_when_stale(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CACHE_DIR", str(tmp_path))
    builds = []
    build = lambda: builds.append(1) or {"n": len(builds)}
    scan_data._cached("thing", False, build)
    assert scan_data._cached("thing", False, build, is_stale=lambda o: False) == {"n": 1}
    assert scan_data._cached("thing", False, build, is_stale=lambda o: True) == {"n": 2}
