"""Entry point: run the gated multi-factor stock scanner.

Reads the latest macro gate snapshot (rebuilding it if it isn't from today),
then scans the S&P 500 only if the gate's zone allows it.

Usage:
    python run_scanner.py            # use today's cached scanner data if present
    python run_scanner.py --refresh  # force a fresh OHLCV + short interest download
    python run_scanner.py --top 50   # rows to print (default 25)
"""

from __future__ import annotations

import argparse

import pandas as pd

from scanner import scan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refresh", action="store_true", help="force a fresh scanner data download")
    parser.add_argument("--top", type=int, default=25, help="number of candidates to print")
    args = parser.parse_args()

    gate = scan.current_gate()
    zone = gate.composite.zone
    print(f"Macro gate: {zone['name']}  (score {gate.composite.score:.1f}, snapshot "
          f"{gate.generated_at:%Y-%m-%d %H:%M})")

    result = scan.run(gate, refresh=args.refresh)
    path = scan.save(result)
    for w in result.warnings:
        print(f"WARNING: {w}")

    if result.status == "disabled":
        print("Scanner DISABLED by macro gate (DEFENSIVE). Candidate list is empty.")
        print(f"Saved to {path}")
        return

    rule = "all stocks ranked" if result.status == "active" else f"composite > {result.min_score:.0f} only"
    print(f"Scanner ACTIVE ({rule}). Data as of {result.as_of:%Y-%m-%d}; "
          f"{result.universe_size} stocks scanned, {len(result.excluded)} excluded (short history); "
          f"short interest data for {result.short_coverage}.")
    print(f"{len(result.candidates)} candidate(s).\n")

    if not result.candidates.empty:
        cols = ["rank", "price", "composite", "momentum_score", "volume_score", "rel_strength_score",
                "high_52w_score", "short_int_score", "crossed", "vol_ratio", "rs_spread_pct",
                "high_ratio", "si_change_pct"]
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(result.candidates[cols].head(args.top).round(2).to_string())
    print(f"\nSaved to {path} (+ scan_candidates.csv)")


if __name__ == "__main__":
    main()
