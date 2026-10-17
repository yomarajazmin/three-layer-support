"""Entry point: refresh market data, recalculate all scores and the backtest, save a snapshot.

Usage:
    python run_macro_gate.py              # force a fresh download
    python run_macro_gate.py --use-cache  # reuse today's cached download if present
"""

from __future__ import annotations

import argparse

import pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--use-cache", action="store_true", help="reuse today's cached market data")
    args = parser.parse_args()

    print("Refreshing market data and recalculating scores..." if not args.use_cache
          else "Recalculating scores (cached data if available)...")
    snap = pipeline.build(refresh=not args.use_cache)
    path = pipeline.save(snap)

    comp = snap.composite
    zone = comp.zone
    print()
    print(f"DEPLOYMENT SCORE  {comp.score:5.1f} / 100   ->  {zone['name']}  ({zone['sizing']:.0%} sizing)")
    print(f"  {zone['guidance']}")
    print(f"  new longs allowed: {zone['allow_new_longs']}   scanner enabled: {zone['scanner_enabled']}")
    print()
    for key, r in comp.results.items():
        print(f"  {r.name:20s} {r.score:5.1f}  w={comp.weights_used[key]:.2f}  {r.value_label}  "
              f"[{r.as_of:%Y-%m-%d}]")
    for key, err in comp.errors.items():
        print(f"  {key:20s}  FAILED  {err}")
    for w in snap.warnings:
        print(f"  WARNING: {w}")

    print()
    if snap.backtest is not None:
        bt = snap.backtest
        start, end = bt.daily.index[0], bt.daily.index[-1]
        print(f"Backtest {start:%Y-%m-%d} -> {end:%Y-%m-%d} ({len(bt.daily)} days, yesterday's score -> today's zone)")
        print(bt.by_zone.round(3).to_string())
        print()
        print(bt.summary.round(2).to_string())
    else:
        print(snap.backtest_error)
    print(f"\nSnapshot saved to {path}")


if __name__ == "__main__":
    main()
