"""Shared synthetic data and fakes for pipeline-level tests (no network, no paid API calls)."""

import copy
import datetime as dt
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

import config
from analyst import analyzer
from analyst.fundamentals import Fundamentals
from data.fetch import MarketData
from signals.composite import zone_for

N_DAYS, N_STOCKS = 1500, 150


def synthetic_market() -> MarketData:
    rng = np.random.default_rng(7)
    idx = pd.bdate_range(end="2026-10-16", periods=N_DAYS)
    walk = lambda drift, vol, shape: 100 * np.cumprod(1 + rng.normal(drift, vol, shape), axis=0)
    tickers = [f"S{i:03d}" for i in range(N_STOCKS)]
    close = pd.DataFrame(walk(0.0004, 0.018, (N_DAYS, N_STOCKS)), index=idx, columns=tickers)
    volume = pd.DataFrame(rng.integers(1_000_000, 5_000_000, (N_DAYS, N_STOCKS)).astype(float),
                          index=idx, columns=tickers)
    vix = 18 + 6 * np.abs(np.sin(np.arange(N_DAYS) / 40)) + rng.normal(0, 0.8, N_DAYS)
    macro = pd.DataFrame({
        config.VIX: vix, config.VIX3M: vix * 1.06 + 0.5,
        config.HYG: walk(0.0001, 0.004, N_DAYS), config.TLT: walk(0.0, 0.009, N_DAYS),
        config.SPY: walk(0.0004, 0.01, N_DAYS),
    }, index=idx)
    return MarketData(macro, close, close * 1.01, volume, tickers, dt.datetime(2026, 10, 16, 18, 0))


def fake_short(tickers, refresh=False):
    rng = np.random.default_rng(len(tickers))
    return pd.DataFrame({"shares_short": rng.uniform(80, 120, len(tickers)), "shares_short_prior": 100.0,
                         "short_date": pd.Timestamp("2026-09-30"), "short_prior_date": pd.Timestamp("2026-08-31"),
                         "short_pct_float": 0.02, "si_error": None}, index=tickers)


def fake_fundamentals(ticker):
    q = [{"quarter_end": d, "revenue": 1000.0, "net_income": 100.0} for d in
         ("2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30")]
    return Fundamentals(ticker, "2026-06-30", q, {"ttm_cfo_to_net_income": 1.2}, [], {"sector": "Tech"})


class FakeClaude:
    def __init__(self):
        self.calls = 0
        self.beta = SimpleNamespace(messages=self)

    def create(self, **kw):
        self.calls += 1
        t = json.loads(kw["messages"][0]["content"].split("\n\n", 1)[1])["ticker"]
        s = 1 + int(t[1:]) % 10
        dim = {"score": s, "rationale": "synthetic"}
        body = {k: dim for k in analyzer.DIMENSIONS} | {"red_flag_items": [], "summary": "synthetic"}
        return SimpleNamespace(stop_reason="end_turn", stop_details=None,
                               content=[SimpleNamespace(type="text", text=json.dumps(body))],
                               usage=SimpleNamespace(input_tokens=1000, output_tokens=500))


def forced(snap, score):
    s = copy.deepcopy(snap)
    s.composite.score, s.composite.zone = score, zone_for(score)
    return s
