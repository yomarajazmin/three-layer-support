import numpy as np
import pandas as pd
import pytest

import config
from signals import breadth, composite, credit_spreads, put_call, vix_level, vix_term_structure
from signals.common import linear_score

IDX = pd.bdate_range("2024-01-01", periods=400)


def series(values) -> pd.Series:
    return pd.Series(values, index=IDX[: len(values)], dtype=float)


# ---- linear_score
@pytest.mark.parametrize("x, expected", [(0.85, 100), (1.0, 50), (1.15, 0), (0.5, 100), (2.0, 0)])
def test_linear_score_term_structure_endpoints(x, expected):
    assert linear_score(x, 0.85, 1.15) == pytest.approx(expected)


# ---- VIX level
def test_vix_level_highest_in_window_scores_zero_and_penalty_floor():
    vix = series(np.linspace(10, 35, 300))  # strictly rising: today is the 1y max
    r = vix_level.compute(vix)
    assert r.history["percentile"].iloc[-1] == pytest.approx(100)
    assert r.score == 0  # 100 - 100 - 10 penalty, clamped


def test_vix_level_lowest_in_window_gets_bonus_clamped():
    vix = series(np.linspace(40, 12, 300))  # strictly falling: today is the 1y min
    r = vix_level.compute(vix)
    pct = 1 / config.TRADING_DAYS_1Y * 100
    assert r.score == pytest.approx(min(100, 100 - pct + 5))


def test_vix_level_needs_full_year():
    with pytest.raises(ValueError):
        vix_level.compute(series(np.full(100, 18.0)))


# ---- Term structure
def test_term_structure_ratio_and_score():
    r = vix_term_structure.compute(series([18.0] * 5), series([20.0] * 5))
    assert r.value == pytest.approx(0.9)
    assert r.score == pytest.approx((1.15 - 0.9) / 0.30 * 100)


# ---- Breadth
def test_breadth_percent_above_sma():
    n = 250
    up = np.linspace(100, 200, n)      # ends above its 200d SMA
    down = np.linspace(200, 100, n)    # ends below
    closes = pd.DataFrame({"A": up, "B": up, "C": up, "D": down}, index=IDX[:n])
    r = breadth.compute(closes)
    assert r.value == pytest.approx(75.0)
    assert r.score == pytest.approx((75 - 30) / 50 * 100)


def test_breadth_ignores_stocks_without_enough_history():
    n = 250
    closes = pd.DataFrame({"A": np.linspace(100, 200, n), "B": np.linspace(200, 100, n)}, index=IDX[:n])
    closes.loc[closes.index[:100], "B"] = np.nan  # B has only 150 days -> no SMA
    r = breadth.compute(closes)
    assert r.value == pytest.approx(100.0)
    assert r.history["n_stocks"].iloc[-1] == 1


# ---- Credit spreads
def test_credit_widening_spike_scores_low():
    rng = np.random.default_rng(0)
    hyg = series(80 + rng.normal(0, 0.1, 300))
    tlt = series(np.full(300, 90.0))
    hyg.iloc[-1] = 75  # HY sells off vs Treasuries -> spread proxy jumps
    r = credit_spreads.compute(hyg, tlt)
    assert r.value > 2
    assert r.score == 0


# ---- Put/Call proxy
@pytest.mark.parametrize("end, expected", [(14.0, 100), (20.0, 62.5), (30.0, 0)])
def test_put_call_roc_mapping(end, expected):
    vix = series([20.0] * 20 + [end])  # ROC over 20 days: -30%, 0%, +50%
    r = put_call.compute(vix)
    assert r.score == pytest.approx(expected)


# ---- Composite
@pytest.mark.parametrize("score, name", [(85, "FULL DEPLOY"), (70, "FULL DEPLOY"), (69.9, "REDUCED"),
                                         (40, "REDUCED"), (39.9, "DEFENSIVE"), (0, "DEFENSIVE")])
def test_zones(score, name):
    assert composite.zone_for(score)["name"] == name


def test_defensive_zone_rules():
    z = composite.zone_for(10)
    assert (z["sizing"], z["allow_new_longs"], z["scanner_enabled"]) == (0.25, False, False)


def test_weights_sum_to_one():
    assert sum(config.SIGNAL_WEIGHTS.values()) == pytest.approx(1.0)


def test_composite_excludes_failed_signal_and_renormalizes():
    vix = series([20.0] * 20 + [14.0])
    ok = lambda: put_call.compute(vix)

    def boom():
        raise RuntimeError("no data")

    res = composite.run({"put_call": ok, "breadth": boom})
    assert res.score == pytest.approx(100)
    assert res.weights_used == {"put_call": pytest.approx(1.0)}
    assert "breadth" in res.errors
