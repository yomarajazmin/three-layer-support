"""Analyst layer tests. Claude is mocked - no paid API calls."""

import datetime as dt
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import config
from analyst import analyzer, blender, fundamentals
from analyst import run as analyst_run
from analyst.analyzer import AnalysisRecord, AnalystScores
from analyst.fundamentals import Fundamentals


def scores_json(eq=8, gt=7, bs=6, mt=5, rf=9, flags=()):
    dim = lambda s: {"score": s, "rationale": f"r{s}"}
    return json.dumps({"earnings_quality": dim(eq), "growth_trajectory": dim(gt), "balance_sheet_health": dim(bs),
                       "margin_trends": dim(mt), "red_flags": dim(rf), "red_flag_items": list(flags),
                       "summary": "ok"})


class FakeMessages:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, SimpleNamespace):
            return reply
        return SimpleNamespace(stop_reason="end_turn", stop_details=None,
                               content=[SimpleNamespace(type="thinking", thinking=""),
                                        SimpleNamespace(type="text", text=reply)],
                               usage=SimpleNamespace(input_tokens=1200, output_tokens=800))


def fake_client(*replies):
    msgs = FakeMessages(replies or [scores_json()])
    return SimpleNamespace(beta=SimpleNamespace(messages=msgs)), msgs


def fake_fetch(quarter_end="2026-06-30"):
    return lambda t: Fundamentals(t, quarter_end, [{"quarter_end": quarter_end, "revenue": 100.0}],
                                  {"ttm_cfo_to_net_income": 1.1}, [], {"sector": "Tech"})


# ---------------------------------------------------------------- scores
def test_fundamental_score_is_mean_times_10():
    s = AnalystScores.model_validate_json(scores_json(8, 7, 6, 5, 9))
    assert s.fundamental_score() == pytest.approx(70.0)


def test_schema_matches_model_fields():
    assert set(analyzer.OUTPUT_SCHEMA["required"]) == set(AnalystScores.model_fields)
    assert analyzer.OUTPUT_SCHEMA["additionalProperties"] is False


def test_score_out_of_range_rejected():
    with pytest.raises(Exception):
        AnalystScores.model_validate_json(scores_json(eq=11))


# ---------------------------------------------------------------- Claude request + cache
def test_request_shape(tmp_path):
    client, msgs = fake_client()
    rec = analyzer.analyze_one("AAA", client, fake_fetch(), db_path=str(tmp_path / "c.sqlite"))
    assert rec.error is None and rec.fundamental_score == pytest.approx(70)
    call = msgs.calls[0]
    assert call["model"] == config.ANALYST_MODEL == "claude-opus-5-5"
    assert call["system"].startswith("You are a senior equity research analyst. Score 1-10 on: Earning Quality")
    assert call["betas"] == ["server-side-fallback-2026-07-01"] and call["fallbacks"] == "default"
    assert call["output_config"]["format"]["schema"] is analyzer.OUTPUT_SCHEMA
    assert call["output_config"]["effort"] == "high"
    assert '"ticker": "AAA"' in call["messages"][0]["content"]


def test_same_quarter_is_cached_new_quarter_is_not(tmp_path):
    db = str(tmp_path / "c.sqlite")
    client, msgs = fake_client()
    first = analyzer.analyze(["AAA", "BBB"], client, fetch=fake_fetch(), db_path=db)
    assert len(msgs.calls) == 2 and not any(r.cached for r in first.values())
    again = analyzer.analyze(["AAA", "BBB"], client, fetch=fake_fetch(), db_path=db)
    assert len(msgs.calls) == 2 and all(r.cached for r in again.values())
    assert again["AAA"].fundamental_score == first["AAA"].fundamental_score
    analyzer.analyze(["AAA"], client, fetch=fake_fetch("2026-09-30"), db_path=db)
    assert len(msgs.calls) == 3   # new quarter -> new analysis


def test_cache_works_without_client(tmp_path):
    db = str(tmp_path / "c.sqlite")
    client, _ = fake_client()
    analyzer.analyze_one("AAA", client, fake_fetch(), db_path=db)
    assert analyzer.analyze_one("AAA", None, fake_fetch(), db_path=db).cached
    miss = analyzer.analyze_one("ZZZ", None, fake_fetch(), db_path=db)
    assert miss.scores is None and "no Claude client" in miss.error


def test_force_bypasses_cache(tmp_path):
    db = str(tmp_path / "c.sqlite")
    client, msgs = fake_client()
    analyzer.analyze_one("AAA", client, fake_fetch(), db_path=db)
    analyzer.analyze_one("AAA", client, fake_fetch(), db_path=db, force=True)
    assert len(msgs.calls) == 2


def test_refusal_is_an_error_not_a_score(tmp_path):
    refusal = SimpleNamespace(stop_reason="refusal", stop_details=SimpleNamespace(category="cyber"),
                              content=[], usage=SimpleNamespace(input_tokens=0, output_tokens=0))
    client, _ = fake_client(refusal)
    rec = analyzer.analyze_one("AAA", client, fake_fetch(), db_path=str(tmp_path / "c.sqlite"))
    assert rec.scores is None and "refusal" in rec.error
    assert analyzer.cache_get("AAA", "2026-06-30", str(tmp_path / "c.sqlite")) is None


def test_invalid_json_is_an_error(tmp_path):
    client, _ = fake_client('{"earnings_quality": 5}')
    rec = analyzer.analyze_one("AAA", client, fake_fetch(), db_path=str(tmp_path / "c.sqlite"))
    assert rec.scores is None and rec.error


# ---------------------------------------------------------------- cache versioning
@pytest.mark.parametrize("change", [
    lambda mp: mp.setattr(config, "ANALYST_MODEL", "claude-sonnet-5-5"),
    lambda mp: mp.setattr(config, "ANALYST_EFFORT", "medium"),
    lambda mp: mp.setattr(analyzer, "SYSTEM_PROMPT", analyzer.SYSTEM_PROMPT + " Be terse."),
    lambda mp: mp.setattr(analyzer, "USER_PREFIX", "Review:\n\n"),
    lambda mp: mp.setattr(fundamentals, "PAYLOAD_VERSION", fundamentals.PAYLOAD_VERSION + 1),
])
def test_changing_analysis_inputs_invalidates_cache(tmp_path, monkeypatch, change):
    db = str(tmp_path / "c.sqlite")
    client, msgs = fake_client()
    analyzer.analyze_one("AAA", client, fake_fetch(), db_path=db)
    v1 = analyzer.analysis_version()
    assert analyzer.analyze_one("AAA", client, fake_fetch(), db_path=db).cached   # unchanged -> free
    change(monkeypatch)
    assert analyzer.analysis_version() != v1
    rec = analyzer.analyze_one("AAA", client, fake_fetch(), db_path=db)
    assert not rec.cached and len(msgs.calls) == 2 and rec.version == analyzer.analysis_version()


def test_old_version_rows_are_kept_and_served_again_when_inputs_revert(tmp_path, monkeypatch):
    db = str(tmp_path / "c.sqlite")
    client, msgs = fake_client()
    analyzer.analyze_one("AAA", client, fake_fetch(), db_path=db)
    with monkeypatch.context() as mp:
        mp.setattr(config, "ANALYST_EFFORT", "max")
        analyzer.analyze_one("AAA", client, fake_fetch(), db_path=db)
    assert len(msgs.calls) == 2
    assert analyzer.analyze_one("AAA", client, fake_fetch(), db_path=db).cached   # original version still there
    assert len(msgs.calls) == 2


def test_unversioned_legacy_cache_is_set_aside(tmp_path):
    import sqlite3
    db = str(tmp_path / "c.sqlite")
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE analyses (ticker TEXT, quarter_end TEXT, model TEXT, created_at TEXT, "
                "fundamentals_json TEXT, result_json TEXT, input_tokens INTEGER, output_tokens INTEGER, "
                "PRIMARY KEY (ticker, quarter_end))")
    con.execute("INSERT INTO analyses VALUES ('AAA', '2026-06-30', 'm', 'x', '{}', ?, 0, 0)", (scores_json(),))
    con.commit(); con.close()
    assert analyzer.cache_get("AAA", "2026-06-30", db) is None          # never served from the old schema
    con = sqlite3.connect(db)
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    legacy = con.execute("SELECT COUNT(*) FROM analyses_unversioned").fetchone()[0]
    con.close()
    assert {"analyses", "analyses_unversioned"} <= tables and legacy == 1


# ---------------------------------------------------------------- fundamentals
def test_fundamentals_ratios_and_yoy_growth(monkeypatch):
    dates = pd.to_datetime(["2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30", "2025-06-30"])
    inc = pd.DataFrame({d: {"Total Revenue": 110.0 if i == 0 else 100.0, "Net Income": 10.0,
                            "Gross Profit": 40.0, "Operating Income": 20.0} for i, d in enumerate(dates)})
    cf = pd.DataFrame({d: {"Operating Cash Flow": 12.0, "Free Cash Flow": 9.0} for d in dates})
    bs = pd.DataFrame({d: {"Total Debt": 50.0, "Stockholders Equity": 100.0,
                           "Accounts Receivable": 130.0 if i == 0 else 100.0} for i, d in enumerate(dates)})
    tk = SimpleNamespace(quarterly_income_stmt=inc, quarterly_cashflow=cf, quarterly_balance_sheet=bs,
                         info={"longName": "Test Co", "sector": "Industrials", "industry": "Tools"})
    monkeypatch.setattr(fundamentals.yf, "Ticker", lambda t: tk)
    f = fundamentals.get_fundamentals("TST")
    assert f.quarter_end == "2026-06-30" and len(f.quarters) == 4
    q = f.quarters[0]
    assert q["gross_margin"] == pytest.approx(40 / 110) and q["operating_margin"] == pytest.approx(20 / 110)
    assert q["debt_to_equity"] == pytest.approx(0.5) and q["roe_quarterly"] == pytest.approx(0.1)
    assert q["cfo_to_net_income"] == pytest.approx(1.2)
    d = f.derived
    assert d["growth_basis"] == "yoy" and d["growth_from_quarter"] == "2025-06-30"
    assert d["ar_growth"] == pytest.approx(0.30) and d["revenue_growth"] == pytest.approx(0.10)
    assert d["ar_growth_minus_revenue_growth"] == pytest.approx(0.20)
    assert d["ttm_cfo_to_net_income"] == pytest.approx(48 / 40)
    assert f.to_payload()["sector"] == "Industrials"


def test_fundamentals_missing_items_are_none(monkeypatch):
    dates = pd.to_datetime(["2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30"])
    inc = pd.DataFrame({d: {"Total Revenue": 100.0, "Net Income": 10.0} for d in dates})
    tk = SimpleNamespace(quarterly_income_stmt=inc, quarterly_cashflow=pd.DataFrame(),
                         quarterly_balance_sheet=pd.DataFrame(), info={})
    monkeypatch.setattr(fundamentals.yf, "Ticker", lambda t: tk)
    f = fundamentals.get_fundamentals("BNK")
    assert "gross_profit" in f.missing and f.quarters[0]["gross_margin"] is None
    assert f.derived["growth_basis"] == "3q_span" and f.derived["ar_growth"] is None


# ---------------------------------------------------------------- blender
def _rec(t, score10):
    s = AnalystScores.model_validate_json(scores_json(score10, score10, score10, score10, score10))
    return AnalysisRecord(t, "2026-06-30", s, {}, "m", cached=False, created_at="x")


def test_blend_weights_and_flags():
    quant = {f"T{i}": 90 - i * 2 for i in range(6)}             # T0 best ... T5 worst
    cands = pd.DataFrame({"rank": range(1, 7), "composite": list(quant.values())}, index=list(quant))
    fund = {"T0": 1, "T1": 5, "T2": 5, "T3": 5, "T4": 5, "T5": 10}  # analyst hates T0, loves T5
    blended, failed = blender.blend(cands, {t: _rec(t, s) for t, s in fund.items()})
    assert failed.empty
    t5 = blended.loc["T5"]
    assert t5["blended_score"] == pytest.approx(0.6 * 80 + 0.4 * 100)
    assert t5["quant_rank"] == 6 and t5["blended_rank"] == 1 and t5["rank_change"] == 5 and t5["flag"] == "upgrade"
    assert blended.loc["T0", "flag"] == "downgrade" and blended.loc["T0", "rank_change"] <= -3
    assert list(blended["blended_rank"]) == sorted(blended["blended_rank"])
    assert set(blended.loc[["T1", "T2", "T3", "T4"], "flag"]) == {"none"}


def test_blend_excludes_failed_from_ranking():
    cands = pd.DataFrame({"rank": [1, 2], "composite": [80.0, 70.0]}, index=["A", "B"])
    bad = AnalysisRecord("A", None, None, None, None, cached=False, created_at=None, error="boom")
    blended, failed = blender.blend(cands, {"A": bad, "B": _rec("B", 5)})
    assert list(blended.index) == ["B"] and blended.loc["B", "quant_rank"] == 1
    assert list(failed["ticker"]) == ["A"] and failed.iloc[0]["error"] == "boom"


# ---------------------------------------------------------------- run gating
def _scan(status, n=0):
    cands = pd.DataFrame({"rank": range(1, n + 1), "composite": np.linspace(90, 80, n)},
                         index=[f"T{i}" for i in range(n)])
    return SimpleNamespace(status=status, candidates=cands, run_at=dt.datetime.now(),
                           gate_zone={"name": "X"}, gate_score=50.0, warnings=[])


def test_disabled_scan_never_calls_claude(monkeypatch):
    monkeypatch.setattr(analyst_run.analyzer, "make_client",
                        lambda: pytest.fail("client must not be created when scanner is disabled"))
    r = analyst_run.run(_scan("disabled"))
    assert r.blended.empty and r.claude_calls == 0 and "disabled" in r.note


def test_run_analyzes_only_top_n(monkeypatch, tmp_path):
    client, msgs = fake_client()
    monkeypatch.setattr(analyst_run.analyzer, "make_client", lambda: client)
    monkeypatch.setattr(analyst_run.analyzer, "get_fundamentals", fake_fetch())
    monkeypatch.setattr(config, "ANALYST_DB", str(tmp_path / "c.sqlite"))
    orig = analyzer.analyze
    monkeypatch.setattr(analyst_run.analyzer, "analyze",
                        lambda tickers, client=None, force=False: orig(tickers, client, force, fake_fetch(),
                                                                       str(tmp_path / "c.sqlite")))
    r = analyst_run.run(_scan("active", 30), top_n=10)
    assert len(msgs.calls) == 10 and len(r.blended) == 10 and r.claude_calls == 10
    assert r.input_tokens == 12000
