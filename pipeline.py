"""End-to-end pipeline: data -> six signals -> composite -> backtest -> snapshot on disk."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

import config
import storage
from backtest import deployment_backtest
from backtest.deployment_backtest import BacktestResult
from data.fetch import MarketData, load_market_data
from data.market_hours import data_is_stale
from signals import (breadth, composite, credit_spreads, crowding, put_call, vix_level,
                     vix_term_structure)
from signals.common import SignalResult
from signals.composite import CompositeResult


@dataclass
class Snapshot:
    generated_at: dt.datetime
    data_downloaded_at: dt.datetime
    composite: CompositeResult
    history: pd.DataFrame            # daily signal scores + composite
    backtest: BacktestResult | None
    backtest_error: str | None
    warnings: list[str] = field(default_factory=list)   # data-quality problems to surface

    @property
    def as_of(self) -> pd.Timestamp:
        """Date of the latest market data used by any signal."""
        return max(r.as_of for r in self.composite.results.values())

    def is_stale(self) -> bool:
        """Data ends before the latest completed session and was downloaded before it closed."""
        return data_is_stale(self.as_of, self.data_downloaded_at)


SCHEMA = storage.fingerprint(Snapshot, CompositeResult, SignalResult, BacktestResult)


def signal_runners(data: MarketData) -> dict:
    vix = lambda: data.series(config.VIX)
    return {
        vix_level.KEY: lambda: vix_level.compute(vix()),
        vix_term_structure.KEY: lambda: vix_term_structure.compute(vix(), data.series(config.VIX3M)),
        breadth.KEY: lambda: breadth.compute(data.sp500),
        credit_spreads.KEY: lambda: credit_spreads.compute(data.series(config.HYG), data.series(config.TLT)),
        put_call.KEY: lambda: put_call.compute(vix()),
        crowding.KEY: lambda: crowding.compute(data.sp500),
    }


def build(refresh: bool = False) -> Snapshot:
    data = load_market_data(refresh=refresh)
    comp = composite.run(signal_runners(data))
    hist = composite.history(comp)
    bt, bt_error = None, None
    if comp.errors:
        bt_error = "Backtest skipped: it requires all six signals ({} failed).".format(", ".join(comp.errors))
    else:
        try:
            bt = deployment_backtest.run(hist["composite"], data.series(config.SPY))
        except Exception as exc:
            bt_error = f"{type(exc).__name__}: {exc}"
    # Signal failures are reported via comp.errors; warnings carry data-quality problems.
    return Snapshot(dt.datetime.now(), data.downloaded_at, comp, hist, bt, bt_error, data.warnings())


def save(snapshot: Snapshot, path: str = config.SNAPSHOT_FILE) -> Path:
    return storage.save(snapshot, path, SCHEMA)


def load(path: str = config.SNAPSHOT_FILE) -> Snapshot | None:
    """Latest snapshot, or None if missing or written by an incompatible version of the code."""
    return storage.load(path, SCHEMA, Snapshot)
