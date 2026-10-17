import numpy as np
import pandas as pd
import pytest

import config
from backtest import deployment_backtest
from signals import crowding

IDX = pd.bdate_range("2020-01-01", periods=1000)


def random_universe(n_stocks=150, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.02, size=(len(IDX), n_stocks))
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=IDX,
                        columns=[f"S{i}" for i in range(n_stocks)])


def test_rebalance_dates_are_month_ends():
    rebal = crowding.rebalance_dates(IDX)
    assert rebal[0] == pd.Timestamp("2020-01-31")
    assert all(d.month != (d + pd.offsets.BDay(1)).month or d == IDX[-1] for d in rebal)


def test_holdings_start_the_day_after_rebalance():
    rebal = pd.DatetimeIndex([IDX[10]])
    members = pd.DataFrame({"A": [True], "B": [False]}, index=rebal)
    held = crowding._held(members, IDX[:15])
    assert not held.loc[IDX[10], "A"]  # formed at this close, not yet held
    assert held.loc[IDX[11], "A"] and held.loc[IDX[14], "A"]
    assert not held["B"].any()


def test_decile_baskets_size():
    closes = random_universe()
    signal = closes.pct_change(20)
    rebal = crowding.rebalance_dates(IDX)[3:4]
    long_m, short_m = crowding._baskets(signal, rebal, long_top=True)
    assert long_m.sum(axis=1).iloc[0] == 15 and short_m.sum(axis=1).iloc[0] == 15
    assert not (long_m & short_m).any().any()


def test_crowding_score_history_bounds():
    h = crowding.score_history(random_universe())
    valid = h.dropna(subset=["score"])
    assert len(valid) > 100
    assert valid["value"].between(-1, 1).all()
    assert valid["score"].between(0, 100).all()
    # first factor return only after the 3y value lookback is available
    assert h.index[0] > IDX[config.CROWDING_VALUE_LOOKBACK]


@pytest.mark.parametrize("corr, expected", [(0.3, 100), (-0.8, 0), (-0.25, 50), (0.9, 100), (-1.0, 0)])
def test_crowding_mapping(corr, expected):
    from signals.common import linear_score
    assert linear_score(corr, config.CROWDING_CORR_AT_100, config.CROWDING_CORR_AT_0) == pytest.approx(expected)


def test_backtest_uses_previous_day_score():
    idx = IDX[:10]
    score = pd.Series(20.0, index=idx)
    score.iloc[5] = 90.0                       # one FULL DEPLOY reading at day 5's close
    spy = pd.Series(100 * 1.01 ** np.arange(10), index=idx)
    bt = deployment_backtest.run(score, spy, days=100)
    assert bt.daily.loc[idx[5], "zone"] == "DEFENSIVE"     # day 5 return uses day 4 score
    assert bt.daily.loc[idx[6], "zone"] == "FULL DEPLOY"   # day 6 return uses day 5 score
    assert bt.daily.loc[idx[6], "strategy_ret"] == pytest.approx(0.01)
    assert bt.daily.loc[idx[7], "strategy_ret"] == pytest.approx(0.0025)
    assert bt.by_zone.loc["FULL DEPLOY", "Days"] == 1
    assert bt.by_zone["Days"].sum() == len(bt.daily) == 9
