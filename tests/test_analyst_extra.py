"""Analyst layer: API error handling, response validation, client setup, fundamentals edge cases,
blender edge cases. Claude is always faked; the SDK's real exception classes are raised."""

import json
from types import SimpleNamespace

import anthropic
import httpx2
import pandas as pd
import pytest

from analyst import analyzer, blender, fundamentals
from analyst import run as analyst_run
from analyst.analyzer import AnalysisRecord, AnalystScores
from analyst.fundamentals import Fundamentals

URL = "https://api.anthropic.com/v1/messages"
REQ = httpx2.Request("POST", URL)


def scores_json(n=7):
    dim = {"score": n, "rationale": "r"}
    return json.dumps({k: dim for k in analyzer.DIMENSIONS} | {"red_flag_items": [], "summary": "s"})


def fetch_ok(t):
    return Fundamentals(t, "2026-06-30", [{"quarter_end": "2026-06-30"}], {}, [], {})


def client_raising(exc):
    def create(**kw):
        raise exc
    return SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))


def client_returning(**resp):
    base = dict(stop_reason="end_turn", stop_details=None, content=[],
                usage=SimpleNamespace(input_tokens=5, output_tokens=6))
    msg = SimpleNamespace(**(base | resp))
    return SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: msg)))


@pytest.mark.parametrize("exc, expected", [
    (anthropic.AuthenticationError("bad key", response=httpx2.Response(401, request=REQ), body=None),
     "invalid ANTHROPIC_API_KEY"),
    (anthropic.RateLimitError("slow down", response=httpx2.Response(429, request=REQ), body=None),
     "rate limited"),
    (anthropic.InternalServerError("boom", response=httpx2.Response(500, request=REQ), body=None),
     "Claude API 500"),
    (anthropic.BadRequestError("bad schema", response=httpx2.Response(400, request=REQ), body=None),
     "Claude API 400"),
    (anthropic.APIConnectionError(request=REQ), "network error"),
])
def test_api_errors_become_error_records_and_are_not_cached(tmp_path, exc, expected):
    db = str(tmp_path / "c.sqlite")
    rec = analyzer.analyze_one("AAA", client_raising(exc), fetch_ok, db_path=db)
    assert rec.scores is None and expected in rec.error
    assert rec.quarter_end == "2026-06-30" and rec.fundamentals is not None     # payload kept for display
    assert analyzer.cache_get("AAA", "2026-06-30", db) is None


def test_max_tokens_and_missing_text_are_errors(tmp_path):
    db = str(tmp_path / "c.sqlite")
    rec = analyzer.analyze_one("A", client_returning(stop_reason="max_tokens"), fetch_ok, db_path=db)
    assert "max_tokens" in rec.error
    rec = analyzer.analyze_one("B", client_returning(content=[SimpleNamespace(type="thinking")]), fetch_ok, db_path=db)
    assert "No text block" in rec.error


def test_fallback_and_thinking_blocks_are_skipped(tmp_path):
    content = [SimpleNamespace(type="fallback"), SimpleNamespace(type="thinking", thinking=""),
               SimpleNamespace(type="text", text=scores_json(9))]
    rec = analyzer.analyze_one("A", client_returning(content=content), fetch_ok, db_path=str(tmp_path / "c.sqlite"))
    assert rec.fundamental_score == 90 and rec.input_tokens == 5 and rec.output_tokens == 6


def test_fundamentals_failure_skips_claude(tmp_path):
    calls = []
    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=lambda **k: calls.append(1))))

    def bad_fetch(t):
        raise RuntimeError("No quarterly income statement")
    rec = analyzer.analyze_one("A", client, bad_fetch, db_path=str(tmp_path / "c.sqlite"))
    assert rec.error.startswith("fundamentals:") and calls == []


def test_make_client_requires_key(monkeypatch):
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        analyzer.make_client()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-real")
    assert isinstance(analyzer.make_client(), anthropic.Anthropic)


def test_run_without_key_uses_cache_only(monkeypatch, tmp_path):
    def no_key():
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")
    monkeypatch.setattr(analyst_run.analyzer, "make_client", no_key)
    seen = {}
    monkeypatch.setattr(analyst_run.analyzer, "analyze",
                        lambda tickers, client=None, force=False: seen.update(client=client) or {})
    cands = pd.DataFrame({"rank": [1], "composite": [80.0]}, index=["A"])
    scan_res = SimpleNamespace(status="active", candidates=cands, run_at=None, gate_zone={}, gate_score=0,
                               warnings=[])
    r = analyst_run.run(scan_res)
    assert seen["client"] is None and "Only cached analyses" in r.note


# ---------------------------------------------------------------- fundamentals edge cases
def _ticker(inc, cf=None, bs=None, info=None):
    t = SimpleNamespace(quarterly_income_stmt=inc, quarterly_cashflow=cf if cf is not None else pd.DataFrame(),
                        quarterly_balance_sheet=bs if bs is not None else pd.DataFrame())
    if isinstance(info, Exception):
        def boom(self):
            raise info
        cls = type("T", (), {"info": property(boom)})
        obj = cls()
        obj.__dict__.update(t.__dict__)
        return obj
    t.info = info or {}
    return t


DATES = pd.to_datetime(["2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30"])


def test_fcf_falls_back_to_cfo_plus_capex(monkeypatch):
    inc = pd.DataFrame({d: {"Total Revenue": 100.0, "Net Income": 10.0} for d in DATES})
    cf = pd.DataFrame({d: {"Operating Cash Flow": 15.0, "Capital Expenditure": -4.0} for d in DATES})
    monkeypatch.setattr(fundamentals.yf, "Ticker", lambda t: _ticker(inc, cf))
    f = fundamentals.get_fundamentals("X")
    assert f.quarters[0]["free_cash_flow"] == 11.0 and f.derived["ttm_free_cash_flow"] == 44.0


def test_fewer_than_four_quarters_has_no_ttm(monkeypatch):
    inc = pd.DataFrame({d: {"Total Revenue": 100.0, "Net Income": 10.0} for d in DATES[:3]})
    monkeypatch.setattr(fundamentals.yf, "Ticker", lambda t: _ticker(inc))
    f = fundamentals.get_fundamentals("X")
    assert len(f.quarters) == 3 and f.derived["ttm_revenue"] is None and not f.derived["ttm_complete"]


def test_empty_statements_raise(monkeypatch):
    monkeypatch.setattr(fundamentals.yf, "Ticker", lambda t: _ticker(pd.DataFrame()))
    with pytest.raises(RuntimeError, match="No quarterly income statement"):
        fundamentals.get_fundamentals("X")
    nan_inc = pd.DataFrame({d: {"Total Revenue": float("nan"), "Net Income": float("nan")} for d in DATES})
    monkeypatch.setattr(fundamentals.yf, "Ticker", lambda t: _ticker(nan_inc))
    with pytest.raises(RuntimeError, match="No usable"):
        fundamentals.get_fundamentals("X")


def test_profile_lookup_failure_is_tolerated(monkeypatch):
    inc = pd.DataFrame({d: {"Total Revenue": 100.0, "Net Income": 10.0} for d in DATES})
    monkeypatch.setattr(fundamentals.yf, "Ticker", lambda t: _ticker(inc, info=ConnectionError("x")))
    f = fundamentals.get_fundamentals("X")
    assert f.profile == {"company_name": None, "sector": None, "industry": None}


def test_ratio_helpers_reject_bad_inputs():
    assert fundamentals._num("abc") is None and fundamentals._num(float("inf")) is None
    assert fundamentals._ratio(1.0, 0.0) is None and fundamentals._growth(5.0, None) is None
    assert fundamentals._growth(-50.0, -100.0) == pytest.approx(0.5)     # growth vs |base|


# ---------------------------------------------------------------- blender edge cases
def _rec(t, n):
    return AnalysisRecord(t, "q", AnalystScores.model_validate_json(scores_json(n)), {}, "m", False, "x")


def test_blend_empty_candidates_returns_typed_empty_frames():
    blended, failed = blender.blend(pd.DataFrame(), {})
    assert blended.empty and "blended_rank" in blended.columns and list(failed.columns) == ["ticker", "quant_score", "error"]


def test_blend_ties_are_deterministic():
    cands = pd.DataFrame({"rank": [1, 2, 3], "composite": [70.0, 70.0, 70.0]}, index=["C", "A", "B"])
    blended, _ = blender.blend(cands, {t: _rec(t, 5) for t in "ABC"})
    assert list(blended.index) == ["A", "B", "C"] and (blended["rank_change"] == 0).all()


def test_flag_threshold_boundary(monkeypatch):
    from analyst import blender as b
    cands = pd.DataFrame({"rank": range(1, 5), "composite": [90.0, 89.0, 88.0, 87.0]}, index=list("ABCD"))
    # D moves from quant #4 to blended #1 (+3); A falls from #1 to #2 (-1)
    blended, _ = b.blend(cands, {"A": _rec("A", 1), "B": _rec("B", 1), "C": _rec("C", 1), "D": _rec("D", 10)})
    assert blended.loc["D", "rank_change"] == 3 and blended.loc["D", "flag"] == "upgrade"
    monkeypatch.setattr(b.config, "RANK_FLAG_THRESHOLD", 4)
    blended, _ = b.blend(cands, {"A": _rec("A", 1), "B": _rec("B", 1), "C": _rec("C", 1), "D": _rec("D", 10)})
    assert blended.loc["D", "flag"] == "none"
