"""Entry point: Claude analyst layer on top of the gated scanner.

Usage:
    python run_analysis.py --scan-and-analyze   # full end-to-end: macro gate -> scanner -> Claude -> blend
    python run_analysis.py                      # analyze the candidates from the last saved scan
Options:
    --top N      candidates to analyze (default from config.ANALYST_TOP_N)
    --refresh    force fresh scanner data downloads (with --scan-and-analyze)
    --force      ignore the SQLite cache and call Claude again for every candidate

Requires ANTHROPIC_API_KEY in .env (cached analyses work without it).
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

import config
from analyst import run as analyst_run
from scanner import scan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scan-and-analyze", action="store_true", help="run gate + scanner before analyzing")
    parser.add_argument("--top", type=int, default=config.ANALYST_TOP_N)
    parser.add_argument("--refresh", action="store_true", help="force fresh scanner data (with --scan-and-analyze)")
    parser.add_argument("--force", action="store_true", help="bypass the analysis cache")
    args = parser.parse_args()

    if args.scan_and_analyze:
        gate = scan.current_gate()
        print(f"Macro gate: {gate.composite.zone['name']} (score {gate.composite.score:.1f})")
        result = scan.run(gate, refresh=args.refresh)
        scan.save(result)
    else:
        result = scan.load()
        if result is None:
            sys.exit("No saved scan. Run with --scan-and-analyze (or run_scanner.py first).")
    print(f"Scanner: {result.status}, {len(result.candidates)} candidate(s) "
          f"(scan {result.run_at:%Y-%m-%d %H:%M})")

    top = min(args.top, len(result.candidates))
    if top:
        print(f"Analyzing top {top} with {config.ANALYST_MODEL} (cached quarters are free)...")
    run = analyst_run.run(result, top_n=args.top, force=args.force)
    path = analyst_run.save(run)

    for w in run.warnings:
        print(f"WARNING: {w}")
    if run.note:
        print(run.note)
    print(f"Claude calls: {run.claude_calls}  cache hits: {run.cache_hits}  "
          f"tokens in/out: {run.input_tokens:,}/{run.output_tokens:,}")
    if not run.blended.empty:
        cols = ["quant_rank", "blended_rank", "rank_change", "flag", "quant_score", "fundamental_score",
                "blended_score"]
        with pd.option_context("display.width", 160):
            print(run.blended[cols].round(1).to_string())
    for _, row in run.failed.iterrows():
        print(f"  {row['ticker']}: FAILED - {row['error']}")
    print(f"\nSaved to {path} (+ analysis_blended.csv)")


if __name__ == "__main__":
    main()
