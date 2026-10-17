"""Analyst run: pick the top scanner candidates, analyze them with Claude, blend, save."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

import config
import storage
from analyst import analyzer, blender
from analyst.analyzer import AnalysisRecord, AnalystScores, Dimension
from scanner.scan import ScanResult


@dataclass
class AnalysisRun:
    run_at: dt.datetime
    scan_run_at: dt.datetime
    gate_zone: dict
    gate_score: float
    scan_status: str
    model: str
    top_n: int
    blended: pd.DataFrame
    failed: pd.DataFrame
    records: dict[str, AnalysisRecord]
    claude_calls: int
    cache_hits: int
    input_tokens: int
    output_tokens: int
    note: str | None = None
    warnings: list[str] = field(default_factory=list)


SCHEMA = storage.fingerprint(AnalysisRun, AnalysisRecord, AnalystScores, Dimension)


def run(scan_result: ScanResult, top_n: int = config.ANALYST_TOP_N, force: bool = False) -> AnalysisRun:
    cands = scan_result.candidates.head(top_n) if not scan_result.candidates.empty else pd.DataFrame()
    note, records = None, {}
    if scan_result.status == "disabled":
        note = "Scanner disabled by the macro gate (DEFENSIVE); nothing to analyze."
    elif cands.empty:
        note = "The scanner surfaced no candidates; nothing to analyze."
    else:
        try:
            client = analyzer.make_client()
        except RuntimeError as exc:
            client, note = None, f"{exc} Only cached analyses can be used."
        records = analyzer.analyze(list(cands.index), client=client, force=force)

    blended, failed = blender.blend(cands, records)
    fresh = [r for r in records.values() if r.scores is not None and not r.cached]
    return AnalysisRun(
        run_at=dt.datetime.now(), scan_run_at=scan_result.run_at, gate_zone=scan_result.gate_zone,
        gate_score=scan_result.gate_score, scan_status=scan_result.status, model=config.ANALYST_MODEL,
        top_n=top_n, blended=blended, failed=failed, records=records,
        claude_calls=len(fresh), cache_hits=sum(1 for r in records.values() if r.cached),
        input_tokens=sum(r.input_tokens for r in fresh), output_tokens=sum(r.output_tokens for r in fresh),
        note=note,
        warnings=list(scan_result.warnings) + ([f"{len(failed)} of {len(cands)} analyses failed."]
                                               if len(failed) else []),
    )


def save(result: AnalysisRun, path: str = config.ANALYSIS_FILE) -> Path:
    p = storage.save(result, path, SCHEMA)
    result.blended.to_csv(p.with_name("analysis_blended.csv"))
    return p


def load(path: str = config.ANALYSIS_FILE) -> AnalysisRun | None:
    """Latest analysis run, or None if missing or written by an incompatible version of the code."""
    return storage.load(path, SCHEMA, AnalysisRun)
