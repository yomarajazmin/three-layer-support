"""End-to-end: macro gate -> scanner -> Claude analyst -> blend, on synthetic market data.

No network: market data, short interest, fundamentals and Claude are all faked.
Runs inside a temp directory so the relative output/ and .cache/ paths never
touch the real ones.
"""

import sys

import numpy as np
import pytest

import config
import pipeline
from analyst import run as analyst_run
from scanner import data as scan_data
from scanner import scan
from synthetic import N_STOCKS, forced

def test_full_pipeline(world):
    # ---- layer 1: macro gate
    snap = pipeline.build()
    assert set(snap.composite.results) == set(config.SIGNAL_WEIGHTS) and not snap.composite.errors
    assert snap.backtest is not None and len(snap.backtest.daily) == config.BACKTEST_DAYS
    assert snap.backtest.by_zone["Days"].sum() == config.BACKTEST_DAYS
    assert 0 <= snap.composite.score <= 100 and snap.warnings == []
    pipeline.save(snap)
    loaded = pipeline.load()
    assert loaded.composite.score == snap.composite.score
    assert (world.tmp / "output" / "snapshot.pkl").exists()

    # ---- layer 2: scanner under each zone
    assert scan.run(forced(snap, 20)).status == "disabled"
    reduced = scan.run(forced(snap, 55))
    assert reduced.status == "filtered" and (reduced.candidates["composite"] > 75).all()
    full = scan.run(forced(snap, 85))
    assert full.status == "active" and len(full.candidates) == full.universe_size == N_STOCKS
    assert full.as_of == world.market.sp500.index[-1] and full.short_coverage == N_STOCKS
    scan.save(full)
    assert scan.load().universe_size == N_STOCKS

    # ---- layer 3: Claude analyst + blend (top 10), then a cached re-run
    run1 = analyst_run.run(full)
    assert world.claude.calls == config.ANALYST_TOP_N == len(run1.blended) == run1.claude_calls
    assert run1.failed.empty and set(run1.blended.index) == set(full.candidates.head(10).index)
    b = run1.blended
    assert np.allclose(b["blended_score"], 0.6 * b["quant_score"] + 0.4 * b["fundamental_score"])
    assert (b["rank_change"] == b["quant_rank"] - b["blended_rank"]).all()
    run2 = analyst_run.run(full)
    assert world.claude.calls == 10 and run2.cache_hits == 10 and run2.claude_calls == 0
    analyst_run.save(run2)
    assert len(analyst_run.load().blended) == 10
    assert (world.tmp / "output" / "analyst_cache.sqlite").exists()


def test_disabled_gate_short_circuits_everything(world, monkeypatch):
    snap = forced(pipeline.build(), 10)
    monkeypatch.setattr(scan_data, "load_market_data", lambda refresh=False: pytest.fail("no download"))
    res = scan.run(snap)
    run = analyst_run.run(res)
    assert res.candidates.empty and run.blended.empty and world.claude.calls == 0


def test_cli_entry_points(world, monkeypatch, capsys):
    import run_analysis
    import run_macro_gate
    import run_scanner
    for mod, argv in ((run_macro_gate, []), (run_scanner, []), (run_analysis, [])):
        monkeypatch.setattr(sys, "argv", [mod.__name__, *argv])
        mod.main()
    out = capsys.readouterr().out
    assert "DEPLOYMENT SCORE" in out and "Backtest" in out and "Macro gate:" in out
    assert "Scanner:" in out and "Saved to" in out
    for f in ("snapshot.pkl", "scan.pkl", "analysis.pkl", "scan_candidates.csv", "analysis_blended.csv"):
        assert (world.tmp / "output" / f).exists()
