"""CLI entry points: argument handling and the branches the end-to-end test doesn't reach."""

import sys
from types import SimpleNamespace

import pandas as pd
import pytest

import run_analysis
import run_macro_gate
import run_scanner


def argv(monkeypatch, mod, *args):
    monkeypatch.setattr(sys, "argv", [mod.__name__, *args])


def test_run_analysis_without_saved_scan_exits(monkeypatch):
    monkeypatch.setattr(run_analysis.scan, "load", lambda: None)
    argv(monkeypatch, run_analysis)
    with pytest.raises(SystemExit, match="No saved scan"):
        run_analysis.main()


def test_run_analysis_flags_are_passed_through(monkeypatch, capsys):
    seen = {}
    res = SimpleNamespace(status="active", candidates=pd.DataFrame({"composite": [1.0]}, index=["A"]),
                          run_at=pd.Timestamp("2026-10-16"))
    monkeypatch.setattr(run_analysis.scan, "load", lambda: res)
    blended = pd.DataFrame(columns=["quant_rank"])
    fake_run = SimpleNamespace(note=None, claude_calls=0, cache_hits=1, input_tokens=0, output_tokens=0,
                               blended=blended, failed=pd.DataFrame({"ticker": ["A"], "error": ["boom"]}),
                               warnings=["low coverage"])
    monkeypatch.setattr(run_analysis.analyst_run, "run",
                        lambda r, top_n, force: seen.update(top_n=top_n, force=force) or fake_run)
    monkeypatch.setattr(run_analysis.analyst_run, "save", lambda r: "output/analysis.pkl")
    argv(monkeypatch, run_analysis, "--top", "3", "--force")
    run_analysis.main()
    out = capsys.readouterr().out
    assert seen == {"top_n": 3, "force": True}
    assert "WARNING: low coverage" in out and "A: FAILED - boom" in out


def test_run_scanner_disabled_branch(monkeypatch, capsys):
    gate = SimpleNamespace(composite=SimpleNamespace(zone={"name": "DEFENSIVE"}, score=12.0),
                           generated_at=pd.Timestamp("2026-10-16 17:00"))
    monkeypatch.setattr(run_scanner.scan, "current_gate", lambda: gate)
    seen = {}
    monkeypatch.setattr(run_scanner.scan, "run", lambda g, refresh: seen.update(refresh=refresh) or
                        SimpleNamespace(status="disabled", warnings=[]))
    monkeypatch.setattr(run_scanner.scan, "save", lambda r: "output/scan.pkl")
    argv(monkeypatch, run_scanner, "--refresh")
    run_scanner.main()
    assert seen["refresh"] is True and "Scanner DISABLED" in capsys.readouterr().out


def test_run_macro_gate_reports_failed_signals_and_warnings(monkeypatch, capsys):
    comp = SimpleNamespace(score=50.0, zone={"name": "REDUCED", "sizing": 0.6, "guidance": "g",
                                             "allow_new_longs": True, "scanner_enabled": True},
                           results={}, weights_used={}, errors={"breadth": "RuntimeError: no data"})
    snap = SimpleNamespace(composite=comp, warnings=["Price download failed"], backtest=None,
                           backtest_error="Backtest skipped")
    seen = {}
    monkeypatch.setattr(run_macro_gate.pipeline, "build", lambda refresh: seen.update(refresh=refresh) or snap)
    monkeypatch.setattr(run_macro_gate.pipeline, "save", lambda s: "output/snapshot.pkl")
    argv(monkeypatch, run_macro_gate, "--use-cache")
    run_macro_gate.main()
    out = capsys.readouterr().out
    assert seen["refresh"] is False
    assert "breadth" in out and "FAILED" in out and "WARNING: Price download failed" in out
    assert "Backtest skipped" in out
