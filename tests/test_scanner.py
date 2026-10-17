import datetime as dt
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import config
from scanner import scan
from scanner.data import ScanData
from scanner.factors import (high_proximity, momentum_crossover, relative_strength, short_interest,
                             volume_surge)

IDX = pd.bdate_range("2025-01-01", periods=250)


def test_crossover_detects_recent_cross_only():
    falling_then_rising = np.r_[np.linspace(200, 100, 240), np.linspace(100, 130, 10)]
    old_cross = np.r_[np.linspace(100, 80, 120), np.linspace(80, 200, 130)]
    always_falling = np.linspace(200, 100, 250)
    close = pd.DataFrame({"NEW": falling_then_rising, "OLD": old_cross, "DOWN": always_falling}, index=IDX)
    out = momentum_crossover.compute(close)
    fast = close.ewm(span=10, adjust=False).mean()
    slow = close.ewm(span=50, adjust=False).mean()
    above = fast["NEW"] > slow["NEW"]
    first_above = above[above & ~above.shift(1, fill_value=False)].index[-1]
    expected_recent = first_above >= IDX[-5]
    assert bool(out.loc["NEW", "crossed"]) == expected_recent
    assert not out.loc["OLD", "crossed"]      # crossed long ago
    assert not out.loc["DOWN", "crossed"]
    assert out.loc["OLD", "score"] == 0 and out.loc["DOWN", "score"] == 0


def test_crossover_scores_rank_among_crossers():
    base = np.r_[np.linspace(101, 100, 247), [108.0, 109.0, 110.0]]   # jump 3 days ago forces a cross
    close = pd.DataFrame({"A": base, "B": np.r_[base[:247], [112.0, 114.0, 116.0]]}, index=IDX)
    out = momentum_crossover.compute(close)
    assert out["crossed"].all(), out
    assert out.loc["B", "score"] == 100 and out.loc["A", "score"] == 50


@pytest.mark.parametrize("ratio, expected", [(0.7, 0), (2.0, 100), (1.35, 50), (3.0, 100), (0.5, 0)])
def test_volume_surge_mapping(ratio, expected):
    # 15 days at 100 then 5 days at v: 20d avg = (1500 + 5v)/20, 5d avg = v
    # => ratio = 20v / (1500 + 5v)  =>  v = 1500r / (20 - 5r)
    v = 1500 * ratio / (20 - 5 * ratio)
    vol = pd.DataFrame({"X": [100.0] * 15 + [v] * 5}, index=IDX[:20])
    out = volume_surge.compute(vol)
    assert out.loc["X", "vol_ratio"] == pytest.approx(ratio)
    assert out.loc["X", "score"] == pytest.approx(expected)


def test_relative_strength_percentile():
    close = pd.DataFrame({"W": np.linspace(100, 90, 30), "S": np.linspace(100, 110, 30),
                          "M": np.linspace(100, 101, 30)}, index=IDX[:30])
    spy = pd.Series(np.linspace(100, 100.5, 30), index=IDX[:30])
    out = relative_strength.compute(close, spy)
    assert list(out["score"].sort_values().index) == ["W", "M", "S"]
    assert out.loc["S", "score"] == 100


@pytest.mark.parametrize("last, expected", [(100, 100), (95, 100), (82.5, 50), (70, 0), (50, 0)])
def test_high_proximity_mapping(last, expected):
    close = pd.DataFrame({"X": [80.0, 100.0, 90.0, last]}, index=IDX[:4])
    assert high_proximity.compute(close).loc["X", "score"] == pytest.approx(expected)


def test_short_interest_decline_ranks_higher_and_missing_is_neutral():
    short = pd.DataFrame({
        "shares_short": [80.0, 100.0, 120.0, np.nan],
        "shares_short_prior": [100.0, 100.0, 100.0, 100.0],
        "short_date": pd.NaT, "short_pct_float": 0.01,
    }, index=["DOWN", "FLAT", "UP", "NA"])
    out = short_interest.compute(short)
    assert out.loc["DOWN", "score"] > out.loc["FLAT", "score"] > out.loc["UP", "score"]
    assert out.loc["NA", "score"] == 50 and not out.loc["NA", "si_has_data"]


def _gate(score):
    from signals.composite import zone_for
    return SimpleNamespace(composite=SimpleNamespace(score=score, zone=zone_for(score)),
                           generated_at=dt.datetime.now(), warnings=[])


def _fake_data(n=6):
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2025-01-01", periods=250)
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0.001, 0.02, (250, n)), axis=0), index=idx,
                         columns=[f"T{i}" for i in range(n)])
    vol = pd.DataFrame(rng.integers(1_000_000, 3_000_000, (250, n)).astype(float), index=idx, columns=close.columns)
    spy = pd.Series(100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 250)), index=idx)
    return ScanData(close, close, vol, spy, dt.datetime.now())


@pytest.fixture
def fake_sources(monkeypatch):
    d = _fake_data()
    calls = {"ohlcv": 0}

    def ohlcv(refresh=False):
        calls["ohlcv"] += 1
        return d

    short = pd.DataFrame({"shares_short": 90.0, "shares_short_prior": 100.0, "short_date": pd.NaT,
                          "short_pct_float": 0.01}, index=d.close.columns)
    monkeypatch.setattr(scan.scan_data, "load_ohlcv", ohlcv)
    monkeypatch.setattr(scan.scan_data, "load_short_interest", lambda t, refresh=False: short.loc[t])
    return calls


def test_defensive_disables_scanner_without_downloading(fake_sources):
    res = scan.run(_gate(20))
    assert res.status == "disabled" and res.candidates.empty
    assert fake_sources["ohlcv"] == 0


def test_full_deploy_returns_all_ranked(fake_sources):
    res = scan.run(_gate(85))
    assert res.status == "active"
    assert len(res.candidates) == len(res.ranked) == 6
    assert res.candidates["composite"].is_monotonic_decreasing
    assert list(res.candidates["rank"]) == [1, 2, 3, 4, 5, 6]


def test_reduced_only_surfaces_above_75(fake_sources):
    res = scan.run(_gate(55))
    assert res.status == "filtered" and res.min_score == config.SCANNER_MIN_SCORE["REDUCED"] == 75
    assert (res.candidates["composite"] > 75).all()
    assert len(res.candidates) == (res.ranked["composite"] > 75).sum()


def test_composite_is_equal_weight_mean(fake_sources):
    res = scan.run(_gate(85))
    cols = [f"{k}_score" for k in scan.FACTORS]
    assert np.allclose(res.ranked["composite"], res.ranked[cols].mean(axis=1))
