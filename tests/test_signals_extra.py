"""Edge cases and behavior of the macro signals, composite, backtest and crowding factors."""

import numpy as np
import pandas as pd
import pytest

import config
from backtest import deployment_backtest
from signals import breadth, composite, credit_spreads, crowding, put_call, vix_level, vix_term_structure
from signals.common import latest, linear_score

IDX = pd.bdate_range("2023-01-02", periods=900)


def s(values, start=0):
    return pd.Series(values, index=IDX[start:start + len(values)], dtype=float)


# ---------------------------------------------------------------- common helpers
def test_linear_score_works_on_series_arrays_and_both_directions():
    x = pd.Series([0.0, 50.0, 100.0, 150.0])
    assert list(linear_score(x, 100, 0)) == [0, 50, 100, 100]          # higher is better
    assert list(linear_score(np.array([0.0, 50.0, 100.0]), 0, 100)) == [100, 50, 0]   # lower is better
    assert np.isnan(linear_score(np.nan, 100, 0))


def test_latest_skips_trailing_nan_and_raises_when_empty():
    h = pd.DataFrame({"score": [10.0, 20.0, np.nan], "value": [1, 2, 3]}, index=IDX[:3])
    assert latest(h).name == IDX[1] and latest(h)["score"] == 20
    with pytest.raises(ValueError):
        latest(pd.DataFrame({"score": [np.nan]}, index=IDX[:1]))


# ---------------------------------------------------------------- VIX level
@pytest.mark.parametrize("n, label", [(1, "1st"), (2, "2nd"), (3, "3rd"), (4, "4th"), (11, "11th"),
                                      (12, "12th"), (13, "13th"), (21, "21st"), (22, "22nd"),
                                      (99.6, "100th"), (101, "101st"), (111, "111th")])
def test_ordinal_suffixes(n, label):
    assert vix_level._ordinal(n) == label


def _vix_with_last(last):
    base = list(np.linspace(16, 40, 251))           # 251 distinct values spanning the range
    return s(base + [last])


@pytest.mark.parametrize("last, adj", [(15.0, 0), (14.99, 5), (30.0, 0), (30.01, -10)])
def test_bonus_and_penalty_thresholds_are_strict(last, adj):
    h = vix_level.score_history(_vix_with_last(last))
    row = h.iloc[-1]
    assert row["score"] == pytest.approx(np.clip(100 - row["percentile"] + adj, 0, 100))


def test_vix_level_score_is_inverse_of_percentile():
    rng = np.random.default_rng(1)
    h = vix_level.score_history(s(18 + rng.normal(0, 2, 400))).dropna()
    mid = h[(h["value"] >= 15) & (h["value"] <= 30)]
    assert np.allclose(mid["score"], 100 - mid["percentile"])
    assert h["score"].between(0, 100).all()


# ---------------------------------------------------------------- term structure / put-call / credit
def test_term_structure_aligns_dates_and_labels_regime():
    vix, vix3m = s([30.0, 33.0, 36.0]), s([30.0, 30.0], start=1)    # overlap on 2 dates only
    h = vix_term_structure.score_history(vix, vix3m)
    assert list(h.index) == list(IDX[1:3]) and list(h["value"]) == pytest.approx([1.1, 1.2])
    r = vix_term_structure.compute(vix, vix3m)
    assert "backwardation" in r.value_label and r.score == 0
    assert "contango" in vix_term_structure.compute(s([18.0]), s([20.0])).value_label


def test_put_call_needs_20_days_of_history():
    with pytest.raises(ValueError):
        put_call.compute(s([20.0] * 20))                 # 20 points -> no 20-day ROC yet
    assert put_call.compute(s([20.0] * 21)).score == pytest.approx(62.5)


def test_credit_spreads_tightening_scores_high_and_needs_a_year():
    hyg = s(np.r_[np.full(300, 80.0) + np.sin(np.arange(300)) * 0.1, [86.0]])   # HY rallies vs TLT
    tlt = s(np.full(301, 90.0))
    r = credit_spreads.compute(hyg, tlt)
    assert r.value < -2 and r.score == 100 and "tighter" in r.value_label
    with pytest.raises(ValueError):
        credit_spreads.compute(hyg.iloc[:200], tlt.iloc[:200])


# ---------------------------------------------------------------- breadth
def test_breadth_clamps_and_counts():
    n = 260
    up = np.linspace(100, 200, n)
    closes = pd.DataFrame({f"U{i}": up for i in range(9)} | {"D": up[::-1]}, index=IDX[:n])
    r = breadth.compute(closes)
    assert r.value == pytest.approx(90.0) and r.score == 100            # 90% > 80% cap
    closes_all_down = pd.DataFrame({f"D{i}": up[::-1] for i in range(4)}, index=IDX[:n])
    assert breadth.compute(closes_all_down).score == 0


def test_breadth_requires_200_days():
    with pytest.raises(ValueError):
        breadth.compute(pd.DataFrame({"A": np.linspace(1, 2, 150)}, index=IDX[:150]))


# ---------------------------------------------------------------- composite
def _fixed(key, score):
    from signals.common import SignalResult
    h = pd.DataFrame({"value": [score], "score": [score]}, index=IDX[:1])
    return lambda: SignalResult(key, key, score, score, "", IDX[0], h, "")


def test_composite_uses_configured_weights():
    runners = {k: _fixed(k, 100.0 if k == "vix_level" else 0.0) for k in config.SIGNAL_WEIGHTS}
    assert composite.run(runners).score == pytest.approx(config.SIGNAL_WEIGHTS["vix_level"] * 100)


def test_composite_renormalizes_after_two_failures():
    def boom():
        raise RuntimeError("x")
    runners = {"vix_level": _fixed("vix_level", 80.0), "breadth": _fixed("breadth", 40.0),
               "crowding": boom, "put_call": boom}
    res = composite.run(runners)
    w_v, w_b = config.SIGNAL_WEIGHTS["vix_level"], config.SIGNAL_WEIGHTS["breadth"]
    assert res.score == pytest.approx((80 * w_v + 40 * w_b) / (w_v + w_b))
    assert sum(res.weights_used.values()) == pytest.approx(1.0) and set(res.errors) == {"crowding", "put_call"}


def test_composite_all_failed_raises():
    def boom():
        raise RuntimeError("x")
    with pytest.raises(RuntimeError, match="All signals failed"):
        composite.run({"vix_level": boom})


def test_zone_for_out_of_range_scores():
    assert composite.zone_for(-5)["name"] == "DEFENSIVE"
    assert composite.zone_for(150)["name"] == "FULL DEPLOY"


def test_history_keeps_only_dates_where_every_signal_scored():
    from signals.common import SignalResult
    mk = lambda k, vals: SignalResult(k, k, 0, 0, "", IDX[0],
                                      pd.DataFrame({"score": vals}, index=IDX[:len(vals)]), "")
    res = composite.CompositeResult(0, {}, {"vix_level": mk("vix_level", [10.0, 20.0, 30.0]),
                                            "breadth": mk("breadth", [np.nan, 50.0, 70.0])},
                                    weights_used={"vix_level": 0.5, "breadth": 0.5})
    h = composite.history(res)
    assert list(h.index) == list(IDX[1:3]) and list(h["composite"]) == [35.0, 50.0]


# ---------------------------------------------------------------- backtest
def test_backtest_zone_stats_and_summary_are_exact():
    idx = IDX[:7]
    score = pd.Series([80, 80, 50, 50, 10, 10, 80], index=idx, dtype=float)
    spy = pd.Series([100, 101, 102.01, 100.99, 99.98, 101.98, 104.0], index=idx)
    bt = deployment_backtest.run(score, spy, days=100)
    rets = spy.pct_change().iloc[1:]
    # day 1-2 use score 80 (FULL), 3-4 use 50 (REDUCED), 5-6 use 10 (DEFENSIVE)
    assert list(bt.daily["zone"]) == ["FULL DEPLOY", "FULL DEPLOY", "REDUCED", "REDUCED", "DEFENSIVE", "DEFENSIVE"]
    assert bt.by_zone.loc["FULL DEPLOY", "Avg daily SPY return %"] == pytest.approx(rets.iloc[:2].mean() * 100)
    assert bt.by_zone.loc["REDUCED", "Hit rate %"] == pytest.approx(0.0)      # both REDUCED days down
    assert bt.by_zone.loc["DEFENSIVE", "Hit rate %"] == pytest.approx(100.0)  # both DEFENSIVE days up
    sized = rets * [1, 1, 0.6, 0.6, 0.25, 0.25]
    assert bt.summary.loc["Zone-sized strategy", "Total return %"] == pytest.approx(((1 + sized).prod() - 1) * 100)
    assert bt.summary.loc["SPY buy & hold", "Total return %"] == pytest.approx((104 / 100 - 1) * 100)


def test_max_drawdown():
    r = pd.Series([0.10, -0.50, 0.20])                   # 1.1 -> 0.55 -> 0.66
    assert deployment_backtest._max_drawdown(r) == pytest.approx(-0.5)


def test_backtest_window_and_no_overlap():
    idx = IDX[:30]
    bt = deployment_backtest.run(pd.Series(50.0, index=idx), pd.Series(np.linspace(100, 110, 30), index=idx), days=10)
    assert len(bt.daily) == 10 and bt.daily.index[-1] == idx[-1]
    with pytest.raises(ValueError):
        deployment_backtest.run(pd.Series(50.0, index=IDX[:5]), pd.Series(1.0, index=IDX[10:15]))


def test_zone_with_no_days_has_nan_stats():
    idx = IDX[:10]
    bt = deployment_backtest.run(pd.Series(90.0, index=idx), pd.Series(np.linspace(100, 110, 10), index=idx))
    assert bt.by_zone.loc["DEFENSIVE", "Days"] == 0 and np.isnan(bt.by_zone.loc["DEFENSIVE", "Hit rate %"])


# ---------------------------------------------------------------- crowding factors
def _trending_universe(n_stocks=120, n_days=900, seed=3):
    """Persistent per-stock drifts: past winners keep winning (momentum works, reversal loses)."""
    rng = np.random.default_rng(seed)
    drift = rng.normal(0, 0.001, n_stocks)
    rets = drift + rng.normal(0, 0.01, (n_days, n_stocks))
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=IDX[:n_days],
                        columns=[f"S{i}" for i in range(n_stocks)])


def test_momentum_factor_earns_positive_returns_when_trends_persist():
    f = crowding.factor_returns(_trending_universe()).dropna()
    assert len(f) > 50
    assert f["momentum"].mean() > 0      # long winners / short losers
    assert f["value"].mean() < 0         # long long-term losers / short winners
    corr = f["momentum"].corr(f["value"])
    assert corr < 0


def test_crowding_needs_enough_stocks():
    with pytest.raises(ValueError):
        crowding.compute(_trending_universe(n_stocks=50))      # < CROWDING_MIN_STOCKS -> no baskets


def test_crowding_score_matches_mapping():
    r = crowding.compute(_trending_universe())
    assert r.score == pytest.approx(float(linear_score(r.value, config.CROWDING_CORR_AT_100,
                                                       config.CROWDING_CORR_AT_0)))
    assert -1 <= r.value <= 1
