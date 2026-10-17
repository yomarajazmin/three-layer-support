"""Gated multi-factor scanner.

The scanner only runs when the macro gate allows it:
  FULL DEPLOY -> all stocks ranked by composite score
  REDUCED     -> only stocks with composite score above 75
  DEFENSIVE   -> scanner disabled, empty list (no data is downloaded)
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

import config
import pipeline
import storage
from data.market_hours import data_is_stale
from scanner import data as scan_data
from scanner.factors import (high_proximity, momentum_crossover, relative_strength, short_interest,
                             volume_surge)

FACTORS = {
    "momentum": momentum_crossover.NAME,
    "volume": volume_surge.NAME,
    "rel_strength": relative_strength.NAME,
    "high_52w": high_proximity.NAME,
    "short_int": short_interest.NAME,
}


@dataclass
class ScanResult:
    run_at: dt.datetime
    gate_zone: dict
    gate_score: float
    gate_generated_at: dt.datetime
    status: str                      # "active" | "filtered" | "disabled"
    min_score: float | None          # composite filter applied (None when disabled)
    ranked: pd.DataFrame             # full ranked universe (empty when disabled)
    candidates: pd.DataFrame         # what the gate allows to surface
    as_of: pd.Timestamp | None
    universe_size: int
    excluded: list[str]              # dropped for insufficient/stale history
    short_coverage: int              # tickers with usable short interest data
    data_downloaded_at: dt.datetime | None
    closes: pd.DataFrame             # daily closes of scanned tickers (for charts)
    warnings: list[str] = field(default_factory=list)   # data-quality problems to surface

    def is_stale(self) -> bool:
        """Prices end before the latest completed session and were downloaded before it closed."""
        return self.as_of is not None and data_is_stale(self.as_of, self.data_downloaded_at)


# Schema covers the result class and the factor set (which defines the ranked table's columns).
SCHEMA = storage.fingerprint(ScanResult, extra=sorted(FACTORS))


def current_gate(rebuild_if_stale: bool = True) -> pipeline.Snapshot:
    """Latest macro gate snapshot; rebuilt when missing, incompatible, or missing the latest
    completed session (the market data cache refreshes itself when needed)."""
    snap = pipeline.load()
    if snap is None or (rebuild_if_stale and snap.is_stale()):
        snap = pipeline.build(refresh=False)
        pipeline.save(snap)
    return snap


def eligible(close: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Tickers with enough history and a close on the latest date."""
    enough = close.notna().sum() >= config.SCANNER_MIN_HISTORY
    current = close.iloc[-1].notna()
    ok = close.columns[enough & current]
    return list(ok), sorted(set(close.columns) - set(ok))


def score_universe(d: scan_data.ScanData, short: pd.DataFrame) -> pd.DataFrame:
    """All five factors + equal-weight composite for every eligible ticker, ranked."""
    tickers, _ = eligible(d.close)
    close, volume = d.close[tickers], d.volume[tickers]
    parts = {
        "momentum": momentum_crossover.compute(close),
        "volume": volume_surge.compute(volume),
        "rel_strength": relative_strength.compute(close, d.spy_close),
        "high_52w": high_proximity.compute(close),
        "short_int": short_interest.compute(short.reindex(tickers)),
    }
    out = pd.DataFrame(index=pd.Index(tickers, name="ticker"))
    out["price"] = close.iloc[-1]
    for key, df in parts.items():
        # A factor that can't be computed for a ticker (e.g. zero volume) contributes 0.
        out[f"{key}_score"] = df["score"].reindex(tickers).fillna(0.0)
        out = out.join(df.drop(columns="score"))
    score_cols = [f"{k}_score" for k in FACTORS]
    out["composite"] = out[score_cols].mean(axis=1)
    out = out.sort_values("composite", ascending=False)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


def run(gate: pipeline.Snapshot, refresh: bool = False) -> ScanResult:
    zone = gate.composite.zone
    min_score = config.SCANNER_MIN_SCORE[zone["name"]]
    base = dict(run_at=dt.datetime.now(), gate_zone=zone, gate_score=gate.composite.score,
                gate_generated_at=gate.generated_at)

    if min_score is None or not zone["scanner_enabled"]:
        return ScanResult(**base, status="disabled", min_score=None, ranked=pd.DataFrame(),
                          candidates=pd.DataFrame(), as_of=None, universe_size=0, excluded=[],
                          short_coverage=0, data_downloaded_at=None,
                          closes=pd.DataFrame(), warnings=list(gate.warnings))

    d = scan_data.load_ohlcv(refresh=refresh)
    tickers, excluded = eligible(d.close)
    short = scan_data.load_short_interest(tickers, refresh=refresh)
    ranked = score_universe(d, short)
    candidates = ranked[ranked["composite"] > min_score] if min_score > 0 else ranked
    warnings = list(d.warnings)
    si_warning = scan_data.short_interest_warning(short, ranked["si_has_data"])
    if si_warning:
        warnings.append(si_warning)
    if len(excluded) > 0.05 * (len(excluded) + len(ranked)):
        warnings.append(f"{len(excluded)} stocks excluded for short or missing recent price history.")
    return ScanResult(**base, status="filtered" if min_score > 0 else "active", min_score=min_score,
                      ranked=ranked, candidates=candidates, as_of=d.close.index[-1],
                      universe_size=len(ranked), excluded=excluded,
                      short_coverage=int(ranked["si_has_data"].sum()),
                      data_downloaded_at=d.downloaded_at, closes=d.close[list(ranked.index)],
                      warnings=warnings)


def save(result: ScanResult, path: str = config.SCAN_FILE) -> Path:
    p = storage.save(result, path, SCHEMA)
    result.candidates.to_csv(p.with_name("scan_candidates.csv"))
    return p


def load(path: str = config.SCAN_FILE) -> ScanResult | None:
    """Latest scan, or None if missing or written by an incompatible version of the code."""
    return storage.load(path, SCHEMA, ScanResult)
