"""Scanner factor edge cases, universe eligibility, scoring and short interest parsing."""

import datetime as dt
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import config
from scanner import data as scan_data
from scanner import scan
from scanner.data import ScanData
from scanner.factors import (high_proximity, momentum_crossover, pct_rank, relative_strength,
                             short_interest, volume_surge)

IDX = pd.bdate_range("2025-01-01", periods=260)


def test_pct_rank_ignores_nan_and_shares_ties():
    r = pct_rank(pd.Series([1.0, 2.0, 2.0, np.nan]))
    assert r.iloc[0] == pytest.approx(100 / 3) and r.iloc[1] == r.iloc[2] == pytest.approx(250 / 3)
    assert np.isnan(r.iloc[3])


# ---------------------------------------------------------------- momentum crossover
def _jump(days_ago: int, n=260):
    """Flat-to-slightly-falling series with a jump `days_ago` trading days before the end."""
    base = np.linspace(101, 100, n)
    base[n - 1 - days_ago:] += 10
    return base


@pytest.mark.parametrize("days_ago, crossed", [(0, True), (2, True), (4, True), (5, False), (20, False)])
def test_cross_must_be_within_last_five_days(days_ago, crossed):
    out = momentum_crossover.compute(pd.DataFrame({"X": _jump(days_ago)}, index=IDX))
    assert bool(out.loc["X", "crossed"]) is crossed
    if crossed:
        assert out.loc["X", "days_since_cross"] == days_ago


def test_cross_that_has_already_reversed_does_not_count():
    px = _jump(3)
    px[-1] = 80.0                                   # crossed 3 days ago, collapsed today
    out = momentum_crossover.compute(pd.DataFrame({"X": px}, index=IDX))
    assert not out.loc["X", "crossed"] and out.loc["X", "score"] == 0


def test_single_crosser_scores_100():
    close = pd.DataFrame({"C": _jump(1), "N": np.linspace(200, 100, 260)}, index=IDX)
    out = momentum_crossover.compute(close)
    assert out.loc["C", "score"] == 100 and out.loc["N", "score"] == 0


# ---------------------------------------------------------------- other factors
def test_volume_surge_zero_volume_is_nan_not_crash():
    out = volume_surge.compute(pd.DataFrame({"Z": np.zeros(30), "A": np.ones(30)}, index=IDX[:30]))
    assert np.isnan(out.loc["Z", "vol_ratio"]) and out.loc["A", "vol_ratio"] == pytest.approx(1.0)


def test_relative_strength_fills_missing_spy_dates():
    close = pd.DataFrame({"A": np.linspace(100, 110, 30)}, index=IDX[:30])
    spy = pd.Series(np.linspace(100, 105, 30), index=IDX[:30]).drop(IDX[29])   # SPY missing the last day
    out = relative_strength.compute(close, spy)
    spy_ret = spy.iloc[-1] / spy.reindex(IDX[:30]).ffill().iloc[-21] - 1
    assert out.loc["A", "rs_spread_pct"] == pytest.approx((110 / close["A"].iloc[-21] - 1 - spy_ret) * 100)


def test_high_proximity_uses_only_last_252_days():
    px = np.r_[[500.0], np.full(259, 100.0)]                  # old spike outside the window
    out = high_proximity.compute(pd.DataFrame({"X": px}, index=IDX))
    assert out.loc["X", "high_ratio"] == 1.0 and out.loc["X", "score"] == 100


def test_short_interest_zero_prior_is_missing():
    short = pd.DataFrame({"shares_short": [10.0], "shares_short_prior": [0.0], "short_date": pd.NaT,
                          "short_pct_float": 0.01}, index=["Z"])
    out = short_interest.compute(short)
    assert not out.loc["Z", "si_has_data"] and out.loc["Z", "score"] == config.SCAN_SHORT_MISSING_SCORE


# ---------------------------------------------------------------- universe + scoring
def test_eligible_excludes_short_history_and_stale_tickers():
    close = pd.DataFrame({"OK": 1.0, "NEW": np.r_[np.full(100, np.nan), np.ones(160)],
                          "DELISTED": np.r_[np.ones(259), [np.nan]]}, index=IDX)
    ok, excluded = scan.eligible(close)
    assert ok == ["OK"] and excluded == ["DELISTED", "NEW"]


def test_score_universe_zero_volume_contributes_zero_and_ranks_are_dense():
    rng = np.random.default_rng(0)
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (260, 4)), axis=0), index=IDX,
                         columns=list("ABCD"))
    vol = pd.DataFrame(1e6, index=IDX, columns=list("ABCD"))
    vol["D"] = 0.0
    d = ScanData(close, close, vol, close.mean(axis=1), dt.datetime.now())
    short = pd.DataFrame({"shares_short": 90.0, "shares_short_prior": 100.0, "short_date": pd.NaT,
                          "short_pct_float": 0.01}, index=list("ABCD"))
    out = scan.score_universe(d, short)
    assert out.loc["D", "volume_score"] == 0
    assert sorted(out["rank"]) == [1, 2, 3, 4] and out["composite"].is_monotonic_decreasing


# ---------------------------------------------------------------- short interest lookups
def test_short_interest_one_parses_yfinance_info(monkeypatch):
    info = {"sharesShort": 120, "sharesShortPriorMonth": 100, "dateShortInterest": 1790726400,
            "sharesShortPreviousMonthDate": 1788134400, "shortPercentOfFloat": 0.02}
    monkeypatch.setattr(scan_data.yf, "Ticker", lambda t: SimpleNamespace(info=info))
    row = scan_data._short_interest_one("AAA")
    assert row["shares_short"] == 120 and row["si_error"] is None
    assert row["short_date"] == pd.Timestamp("2026-09-30") and row["short_prior_date"] == pd.Timestamp("2026-08-31")


def test_short_interest_one_records_request_errors(monkeypatch):
    class Boom:
        @property
        def info(self):
            raise ConnectionError("rate limited")
    monkeypatch.setattr(scan_data.yf, "Ticker", lambda t: Boom())
    row = scan_data._short_interest_one("AAA")
    assert row["si_error"] == "ConnectionError" and row["shares_short"] is None and pd.isna(row["short_date"])


def test_load_short_interest_adds_new_universe_members(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CACHE_DIR", str(tmp_path))
    seen = []

    def one(t):
        seen.append(t)
        return {"ticker": t, "shares_short": 1.0, "shares_short_prior": 1.0, "short_date": pd.NaT,
                "short_prior_date": pd.NaT, "short_pct_float": 0.0, "si_error": None}
    monkeypatch.setattr(scan_data, "_short_interest_one", one)
    scan_data.load_short_interest(["A", "B"])
    out = scan_data.load_short_interest(["B", "A", "C"])        # C joined the index since the cache
    assert list(out.index) == ["B", "A", "C"] and seen == ["A", "B", "C"]
