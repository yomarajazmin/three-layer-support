"""Data-quality warnings, staleness detection, shared data slicing and versioned storage."""

import dataclasses
import datetime as dt
import pickle

import numpy as np
import pandas as pd
import pytest

import config
import storage
from data import fetch
from data.market_hours import TZ, data_is_stale, last_completed_session
from scanner import data as scan_data


def et(s):
    return pd.Timestamp(s, tz=TZ)


# ---------------------------------------------------------------- staleness
@pytest.mark.parametrize("now, expected", [
    ("2026-10-16 15:00", "2026-10-15"),   # Fri before cutoff -> Thu
    ("2026-10-16 17:00", "2026-10-16"),   # Fri after cutoff
    ("2026-10-17 12:00", "2026-10-16"),   # Sat -> Fri
    ("2026-10-19 09:00", "2026-10-16"),   # Mon pre-open -> Fri
])
def test_last_completed_session(now, expected):
    assert last_completed_session(et(now)) == dt.date.fromisoformat(expected)


@pytest.mark.parametrize("as_of, downloaded, now, stale", [
    ("2026-10-16", "2026-10-16 17:00", "2026-10-17 12:00", False),   # has Friday's bar
    ("2026-10-15", "2026-10-16 11:00", "2026-10-16 18:00", True),    # downloaded mid-session Friday
    ("2026-10-15", "2026-10-15 17:00", "2026-10-19 18:00", True),    # days old
    ("2026-10-15", "2026-10-16 18:00", "2026-10-16 19:00", False),   # post-close download w/o bar = holiday
])
def test_data_is_stale(as_of, downloaded, now, stale):
    assert data_is_stale(pd.Timestamp(as_of), et(downloaded), now=et(now)) is stale


# ---------------------------------------------------------------- warnings
def market(n_members=100, missing=(), macro_cols=tuple(config.MACRO_TICKERS)):
    idx = pd.bdate_range(end="2026-10-16", periods=400)
    members = [f"S{i}" for i in range(n_members)]
    have = [m for m in members if m not in missing]
    px = pd.DataFrame(100.0, index=idx, columns=have)
    macro = pd.DataFrame(20.0, index=idx, columns=list(macro_cols))
    return fetch.MarketData(macro, px, px * 1.01, px * 1000, members, dt.datetime.now(),
                            missing_tickers=sorted(missing))


def test_no_warnings_for_complete_data():
    assert market().warnings() == []


def test_warns_on_missing_member_prices():
    w = market(missing=[f"S{i}" for i in range(5)]).warnings()
    assert len(w) == 1 and "5 of 100" in w[0]


def test_small_missing_count_below_threshold_is_quiet():
    assert market(missing=["S1"]).warnings() == []      # 1% < 2% threshold


def test_warns_on_missing_macro_series():
    w = market(macro_cols=[t for t in config.MACRO_TICKERS if t != config.HYG]).warnings()
    assert any(config.HYG in x for x in w)


def test_short_interest_warning():
    short = pd.DataFrame({"si_error": ["YFRateLimitError"] * 30 + [None] * 70})
    has = pd.Series([False] * 30 + [True] * 70)
    msg = scan_data.short_interest_warning(short, has)
    assert "70 of 100" in msg and "30 request errors" in msg
    assert scan_data.short_interest_warning(short.iloc[25:], has.iloc[25:]) is None   # 70/75 = 93% covered


def test_short_interest_cache_with_many_errors_is_retried(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CACHE_DIR", str(tmp_path))
    calls = {"n": 0}

    def fake_one(t):
        calls["n"] += 1
        failing = calls["n"] <= 10           # first pass: everything rate-limited
        return {"ticker": t, "shares_short": None if failing else 90, "shares_short_prior": 100,
                "short_date": pd.NaT, "short_prior_date": pd.NaT, "short_pct_float": 0.01,
                "si_error": "YFRateLimitError" if failing else None}
    monkeypatch.setattr(scan_data, "_short_interest_one", fake_one)
    tickers = [f"T{i}" for i in range(10)]
    first = scan_data.load_short_interest(tickers)
    assert first["si_error"].notna().all()
    second = scan_data.load_short_interest(tickers)           # bad cache -> fetched again
    assert second["si_error"].isna().all() and calls["n"] == 20
    scan_data.load_short_interest(tickers)                    # good cache -> reused
    assert calls["n"] == 20


def test_scanner_slices_one_year_from_shared_download():
    m = market()
    m.macro[config.SPY] = np.arange(len(m.macro), dtype=float)
    d = scan_data.from_market(m)
    span = (d.close.index[-1] - d.close.index[0]).days
    assert span <= config.SCANNER_LOOKBACK_DAYS and span > 360
    assert d.close.index.equals(d.volume.index) and d.spy_close.index[-1] == d.close.index[-1]
    assert d.close.index[-1] == m.sp500.index[-1]


# ---------------------------------------------------------------- storage
@dataclasses.dataclass
class Thing:
    a: int
    b: str


def test_storage_roundtrip(tmp_path):
    path = str(tmp_path / "x.pkl")
    schema = storage.fingerprint(Thing)
    storage.save(Thing(1, "x"), path, schema)
    assert storage.load(path, schema, Thing) == Thing(1, "x")
    assert not (tmp_path / "x.pkl.tmp").exists()


def test_storage_rejects_other_schema_type_and_garbage(tmp_path):
    path = str(tmp_path / "x.pkl")
    storage.save(Thing(1, "x"), path, "old-schema")
    assert storage.load(path, storage.fingerprint(Thing), Thing) is None          # schema changed
    storage.save({"not": "a Thing"}, path, "s")
    assert storage.load(path, "s", Thing) is None                                 # wrong type
    with open(path, "wb") as fh:
        pickle.dump(Thing(1, "x"), fh)                                            # legacy raw pickle
    assert storage.load(path, "s", Thing) is None
    (tmp_path / "x.pkl").write_bytes(b"\x80\x05garbage")
    assert storage.load(path, "s", Thing) is None                                 # unreadable
    assert storage.load(str(tmp_path / "missing.pkl"), "s", Thing) is None


def test_fingerprint_changes_with_fields():
    @dataclasses.dataclass
    class V1:
        a: int

    @dataclasses.dataclass
    class V2:
        a: int
        new_field: float = 0.0
    V2.__qualname__ = V1.__qualname__
    assert storage.fingerprint(V1) != storage.fingerprint(V2)
    assert storage.fingerprint(V1, extra=["x"]) != storage.fingerprint(V1, extra=["y"])


def test_real_result_schemas_are_stable_and_distinct():
    import pipeline
    from analyst import run as analyst_run
    from scanner import scan
    assert len({pipeline.SCHEMA, scan.SCHEMA, analyst_run.SCHEMA}) == 3
    assert pipeline.SCHEMA == storage.fingerprint(pipeline.Snapshot, pipeline.CompositeResult,
                                                  pipeline.SignalResult, pipeline.BacktestResult)


def test_current_gate_rebuilds_stale_snapshot(monkeypatch):
    from types import SimpleNamespace

    from scanner import scan
    fresh = SimpleNamespace(is_stale=lambda: False, name="fresh")
    stale = SimpleNamespace(is_stale=lambda: True, name="stale")
    saved = []
    monkeypatch.setattr(scan.pipeline, "build", lambda refresh=False: fresh)
    monkeypatch.setattr(scan.pipeline, "save", lambda s: saved.append(s))
    monkeypatch.setattr(scan.pipeline, "load", lambda: stale)
    assert scan.current_gate().name == "fresh" and saved == [fresh]
    monkeypatch.setattr(scan.pipeline, "load", lambda: None)               # missing / incompatible
    assert scan.current_gate().name == "fresh"
    monkeypatch.setattr(scan.pipeline, "load", lambda: fresh)
    saved.clear()
    assert scan.current_gate().name == "fresh" and saved == []             # reused
