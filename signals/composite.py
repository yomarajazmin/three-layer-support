"""Composite: weighted blend of the six signal scores, mapped to a deployment zone."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

import config
from signals.common import SignalResult


@dataclass
class CompositeResult:
    score: float
    zone: dict
    results: dict[str, SignalResult]
    errors: dict[str, str] = field(default_factory=dict)
    weights_used: dict[str, float] = field(default_factory=dict)


def zone_for(score: float) -> dict:
    for zone in config.ZONES:
        if score >= zone["min_score"]:
            return zone
    return config.ZONES[-1]


def _normalized_weights(keys) -> dict[str, float]:
    raw = {k: config.SIGNAL_WEIGHTS[k] for k in keys}
    total = sum(raw.values())
    return {k: w / total for k, w in raw.items()}


def run(signal_runners: dict[str, Callable[[], SignalResult]]) -> CompositeResult:
    """Run each signal; a failing signal is reported and excluded, and the
    remaining weights are re-normalized so they still sum to 1."""
    results, errors = {}, {}
    for key, runner in signal_runners.items():
        try:
            results[key] = runner()
        except Exception as exc:  # surface any data/compute failure on the dashboard
            errors[key] = f"{type(exc).__name__}: {exc}"
    if not results:
        raise RuntimeError(f"All signals failed: {errors}")
    weights = _normalized_weights(results)
    score = sum(results[k].score * w for k, w in weights.items())
    return CompositeResult(score, zone_for(score), results, errors, weights)


def history(composite: CompositeResult) -> pd.DataFrame:
    """Daily score of each available signal plus the weighted composite.
    Only dates on which every available signal has a score are kept."""
    scores = pd.concat({k: r.history["score"] for k, r in composite.results.items()}, axis=1, sort=True).dropna()
    weights = pd.Series(composite.weights_used)
    scores["composite"] = scores[weights.index].mul(weights).sum(axis=1)
    return scores
