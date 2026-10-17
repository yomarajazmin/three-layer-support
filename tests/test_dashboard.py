"""Dashboard pages rendered headlessly with Streamlit's AppTest, against synthetic snapshots.

Each test builds the gate / scan / analysis snapshots from synthetic data (see the
`world` fixture) inside a temp cwd, then runs one page script and inspects what it rendered.
"""

import dataclasses
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import pipeline
from analyst import run as analyst_run
from data import market_hours
from scanner import scan
from synthetic import forced

ROOT = Path(__file__).resolve().parents[1]
PAGE = {name: str(ROOT / "views" / f"{name}.py") for name in ("macro_gate", "scanner_page", "analyst_page")}


@pytest.fixture
def app(world, monkeypatch):
    st.cache_resource.clear()          # loaders are cached by file mtime across tests
    # "now" = Saturday after the synthetic data's last session, so nothing is stale by default
    monkeypatch.setattr(market_hours, "now_et", lambda: pd.Timestamp("2026-10-17 12:00", tz=market_hours.TZ))
    snap = pipeline.build()
    pipeline.save(snap)

    def run(page):
        at = AppTest.from_file(PAGE[page], default_timeout=60)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        return at
    world.snap, world.run = snap, run
    return world


def texts(elements):
    return " ".join(str(e.value) for e in elements)


# ---------------------------------------------------------------- page 1: macro gate
def test_macro_gate_page_renders_score_signals_and_backtest(app):
    at = app.run("macro_gate")
    assert at.title[0].value == "Market Deployment Gate"
    md = texts(at.markdown)
    assert f"{app.snap.composite.score:.0f}<small> / 100</small>" in md
    assert app.snap.composite.zone["name"] in md
    for r in app.snap.composite.results.values():
        assert f"**{r.name}**" in md
    assert len(at.dataframe) >= 2          # zone stats + summary
    assert not at.warning


def test_macro_gate_page_warns_when_stale_and_on_data_problems(app, monkeypatch):
    pipeline.save(dataclasses.replace(app.snap, warnings=["Price download failed for 12 of 500"]))
    monkeypatch.setattr(market_hours, "now_et", lambda: pd.Timestamp("2026-10-21 18:00", tz=market_hours.TZ))
    at = app.run("macro_gate")
    w = texts(at.warning)
    assert "before the latest completed session" in w and "Price download failed" in w


# ---------------------------------------------------------------- page 2: scanner
def test_scanner_page_full_deploy_table_and_detail(app):
    scan.save(scan.run(forced(app.snap, 85)))
    at = app.run("scanner_page")
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Stocks scanned"] == "150" and metrics["Candidates"] == "150"
    assert len(at.dataframe) == 1 and len(at.dataframe[0].value) == 25          # top 25 by default
    assert at.selectbox[0].value == at.dataframe[0].value["ticker"].iloc[0]       # detail = top pick
    assert "ACTIVE" in texts(at.markdown)
    # the scan used a forced gate, so the page must flag the mismatch with the current gate
    assert "Run the scan again" in texts(at.warning)


def test_scanner_page_filters(app):
    res = scan.run(forced(app.snap, 85))
    scan.save(res)
    n_crossers = int(res.candidates["crossed"].sum())
    assert n_crossers > 0                                    # synthetic data has fresh crossovers
    at = app.run("scanner_page")
    [c for c in at.checkbox if c.label == "Only fresh EMA crossovers"][0].check().run()
    table = at.dataframe[0].value
    assert table["crossed"].all() and len(table) == min(n_crossers, 25)
    at.slider[0].set_value(100.0).run()                      # minimum composite 100 -> nothing passes
    assert "No stocks pass the current filters" in texts(at.info)


def test_scanner_page_disabled(app):
    scan.save(scan.run(forced(app.snap, 10)))
    at = app.run("scanner_page")
    assert "DISABLED" in texts(at.markdown) and len(at.dataframe[0].value) == 0


def test_scanner_page_without_scan_prompts_to_run(app):
    if app.snap.composite.zone["scanner_enabled"]:
        at = app.run("scanner_page")
        assert "No scan yet" in texts(at.info)
    else:
        at = app.run("scanner_page")                           # DEFENSIVE: creates the empty scan itself
        assert scan.load().status == "disabled"


# ---------------------------------------------------------------- page 3: AI analyst
def test_analyst_page_blended_table_glow_and_notes(app):
    res = scan.run(forced(app.snap, 85))
    scan.save(res)
    run = analyst_run.run(res)
    analyst_run.save(run)
    at = app.run("analyst_page")
    md = texts(at.markdown)
    assert "<table class='blend'>" in md and md.count("<tr class=") == 10
    ups, downs = (run.blended["flag"] == "upgrade").sum(), (run.blended["flag"] == "downgrade").sum()
    assert md.count("<tr class='up'>") == ups and md.count("<tr class='down'>") == downs
    assert len(at.expander) == 10
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Analyzed"] == "10" and metrics["Upgraded ≥3"] == str(ups)
    assert "ANTHROPIC_API_KEY not found" in texts(at.sidebar.warning)


def test_analyst_page_button_reuses_cached_analyses(app):
    res = scan.run(forced(app.snap, 85))
    scan.save(res)
    analyst_run.save(analyst_run.run(res))
    calls_before = app.claude.calls
    at = app.run("analyst_page")
    [b for b in at.sidebar.button if b.label == "Analyze last scan"][0].click().run()
    assert not at.exception
    assert app.claude.calls == calls_before                    # same quarters -> all cache hits
    assert analyst_run.load().cache_hits == 10


def test_analyst_page_without_analysis(app):
    at = app.run("analyst_page")
    assert "No analysis yet" in texts(at.info)


def test_app_entry_point_renders_default_page(app):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title[0].value == "Market Deployment Gate"      # default page
