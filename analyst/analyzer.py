"""Candidate analyzer: Claude scores each scanner candidate on fundamental quality.

This is the nondeterministic layer. Results are cached in SQLite keyed by
(ticker, quarter_end, version), so re-analyzing a ticker whose latest reported
quarter hasn't changed costs nothing. `version` is a hash of everything that
shapes the analysis (model, effort, prompts, output schema, payload format), so
changing any of them invalidates the old analyses instead of silently reusing them.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sqlite3
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

import config
from analyst import fundamentals as fundamentals_mod
from analyst.fundamentals import Fundamentals, get_fundamentals

SYSTEM_PROMPT = """You are a senior equity research analyst. Score 1-10 on: Earning Quality, Growth Trajectory, Balance Sheet Health, Margin Trends, red flags. Output as JSON.

Scoring guidance:
- Every score is an integer 1-10 where 10 is the best outcome for a long investor.
- Red Flags: 10 = no red flags found, 1 = severe red flags. List each red flag you find.
- Earnings Quality: consider cash conversion (CFO / net income) and whether receivables are growing faster than revenue.
- Interpret metrics in the context of the company's sector (e.g. banks and insurers have no gross margin, and their operating cash flow and receivables behave differently from industrial companies).
- Base every judgment only on the data provided. Do not use outside knowledge of recent events, and do not invent figures. If data needed for a dimension is missing, say so in that dimension's rationale and score conservatively.
- Ratios are decimals (0.25 = 25%). Monetary values are in the company's reporting currency."""

Score = Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


class Dimension(BaseModel):
    score: Score
    rationale: str


class AnalystScores(BaseModel):
    earnings_quality: Dimension
    growth_trajectory: Dimension
    balance_sheet_health: Dimension
    margin_trends: Dimension
    red_flags: Dimension = Field(description="10 = no red flags, 1 = severe")
    red_flag_items: list[str]
    summary: str

    def fundamental_score(self) -> float:
        """Mean of the five 1-10 scores x 10 -> 10..100."""
        dims = (self.earnings_quality, self.growth_trajectory, self.balance_sheet_health,
                self.margin_trends, self.red_flags)
        return sum(d.score for d in dims) / len(dims) * 10.0


DIMENSIONS = {
    "earnings_quality": "Earnings Quality",
    "growth_trajectory": "Growth Trajectory",
    "balance_sheet_health": "Balance Sheet Health",
    "margin_trends": "Margin Trends",
    "red_flags": "Red Flags (10 = none)",
}

_DIMENSION_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "enum": list(range(1, 11))},
        "rationale": {"type": "string"},
    },
    "required": ["score", "rationale"],
    "additionalProperties": False,
}
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        **{k: _DIMENSION_SCHEMA for k in DIMENSIONS},
        "red_flag_items": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": [*DIMENSIONS, "red_flag_items", "summary"],
    "additionalProperties": False,
}


USER_PREFIX = "Analyze this company's last four quarters of fundamentals:\n\n"


def analysis_version() -> str:
    """Short hash of every input that shapes Claude's analysis, other than the company data."""
    spec = {
        "model": config.ANALYST_MODEL,
        "effort": config.ANALYST_EFFORT,
        "system": SYSTEM_PROMPT,
        "user_prefix": USER_PREFIX,
        "schema": OUTPUT_SCHEMA,
        "quarters": config.ANALYST_QUARTERS,
        "payload_version": fundamentals_mod.PAYLOAD_VERSION,
    }
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]


@dataclass
class AnalysisRecord:
    ticker: str
    quarter_end: str | None
    scores: AnalystScores | None
    fundamentals: dict | None        # payload sent to Claude
    model: str | None
    cached: bool
    created_at: str | None
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    version: str | None = None

    @property
    def fundamental_score(self) -> float | None:
        return self.scores.fundamental_score() if self.scores else None


# ---------------------------------------------------------------- SQLite cache
def _db(path: str = config.ANALYST_DB) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    cols = [r[1] for r in con.execute("PRAGMA table_info(analyses)")]
    if cols and "version" not in cols:   # pre-versioning cache: keep it, but never serve from it
        con.execute("ALTER TABLE analyses RENAME TO analyses_unversioned")
        con.commit()
    con.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            ticker TEXT NOT NULL,
            quarter_end TEXT NOT NULL,
            version TEXT NOT NULL,
            model TEXT NOT NULL,
            created_at TEXT NOT NULL,
            fundamentals_json TEXT NOT NULL,
            result_json TEXT NOT NULL,
            input_tokens INTEGER,
            output_tokens INTEGER,
            PRIMARY KEY (ticker, quarter_end, version)
        )""")
    return con


def cache_get(ticker: str, quarter_end: str, path: str = config.ANALYST_DB,
              version: str | None = None) -> AnalysisRecord | None:
    version = version or analysis_version()
    with closing(_db(path)) as con:
        row = con.execute("SELECT model, created_at, fundamentals_json, result_json FROM analyses "
                          "WHERE ticker = ? AND quarter_end = ? AND version = ?",
                          (ticker, quarter_end, version)).fetchone()
    if row is None:
        return None
    model, created_at, fjson, rjson = row
    return AnalysisRecord(ticker, quarter_end, AnalystScores.model_validate_json(rjson), json.loads(fjson),
                          model, cached=True, created_at=created_at, version=version)


def cache_put(rec: AnalysisRecord, path: str = config.ANALYST_DB) -> None:
    with closing(_db(path)) as con, con:
        con.execute("INSERT OR REPLACE INTO analyses (ticker, quarter_end, version, model, created_at, "
                    "fundamentals_json, result_json, input_tokens, output_tokens) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (rec.ticker, rec.quarter_end, rec.version or analysis_version(), rec.model, rec.created_at,
                     json.dumps(rec.fundamentals), rec.scores.model_dump_json(), rec.input_tokens,
                     rec.output_tokens))


# ---------------------------------------------------------------- Claude call
def make_client() -> anthropic.Anthropic:
    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to a .env file in the project root.")
    return anthropic.Anthropic()


def call_claude(client: anthropic.Anthropic, payload: dict) -> tuple[AnalystScores, int, int]:
    response = client.beta.messages.create(
        model=config.ANALYST_MODEL,
        max_tokens=config.ANALYST_MAX_TOKENS,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",   # on a safety-classifier decline, re-run on Anthropic's recommended model
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": USER_PREFIX + json.dumps(payload, indent=2)}],
        output_config={"effort": config.ANALYST_EFFORT,
                       "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
    )
    if response.stop_reason == "refusal":
        category = getattr(response.stop_details, "category", None) if response.stop_details else None
        raise RuntimeError(f"Claude declined the request (refusal, category={category})")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("Claude response hit max_tokens before completing the JSON")
    text = next((b.text for b in response.content if b.type == "text"), None)
    if text is None:
        raise RuntimeError(f"No text block in response (stop_reason={response.stop_reason})")
    return AnalystScores.model_validate_json(text), response.usage.input_tokens, response.usage.output_tokens


def _error(ticker: str, msg: str, quarter_end: str | None = None, payload: dict | None = None) -> AnalysisRecord:
    return AnalysisRecord(ticker, quarter_end, None, payload, None, cached=False, created_at=None, error=msg)


def analyze_one(ticker: str, client: anthropic.Anthropic | None,
                fetch: Callable[[str], Fundamentals] | None = None,
                db_path: str | None = None, force: bool = False) -> AnalysisRecord:
    fetch = fetch or get_fundamentals          # resolved at call time (patchable in tests)
    db_path = db_path or config.ANALYST_DB
    try:
        f = fetch(ticker)
    except Exception as exc:
        return _error(ticker, f"fundamentals: {type(exc).__name__}: {exc}")
    if not force:
        hit = cache_get(ticker, f.quarter_end, db_path)
        if hit is not None:
            return hit
    payload = f.to_payload()
    if client is None:
        return _error(ticker, "not cached and no Claude client available", f.quarter_end, payload)
    try:
        scores, in_tok, out_tok = call_claude(client, payload)
    except anthropic.AuthenticationError:
        return _error(ticker, "Claude API: invalid ANTHROPIC_API_KEY", f.quarter_end, payload)
    except anthropic.RateLimitError:
        return _error(ticker, "Claude API: rate limited (after SDK retries)", f.quarter_end, payload)
    except anthropic.APIStatusError as exc:
        return _error(ticker, f"Claude API {exc.status_code}: {exc.message}", f.quarter_end, payload)
    except anthropic.APIConnectionError:
        return _error(ticker, "Claude API: network error", f.quarter_end, payload)
    except Exception as exc:  # refusal / truncation / schema validation
        return _error(ticker, f"{type(exc).__name__}: {exc}", f.quarter_end, payload)
    rec = AnalysisRecord(ticker, f.quarter_end, scores, payload, config.ANALYST_MODEL, cached=False,
                         created_at=dt.datetime.now().isoformat(timespec="seconds"),
                         input_tokens=in_tok, output_tokens=out_tok, version=analysis_version())
    cache_put(rec, db_path)
    return rec


def analyze(tickers: list[str], client: anthropic.Anthropic | None = None, force: bool = False,
            fetch: Callable[[str], Fundamentals] | None = None,
            db_path: str | None = None) -> dict[str, AnalysisRecord]:
    """Analyze tickers in parallel; cached (ticker, quarter_end) pairs never call Claude."""
    with ThreadPoolExecutor(max_workers=config.ANALYST_WORKERS) as pool:
        recs = list(pool.map(lambda t: analyze_one(t, client, fetch, db_path, force), tickers))
    return {r.ticker: r for r in recs}
