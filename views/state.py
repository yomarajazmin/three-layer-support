"""Cached loaders for the snapshots written by run_macro_gate.py and run_scanner.py."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

import config
import pipeline
from scanner import scan


def _mtime(path: str) -> float | None:
    p = Path(path)
    return p.stat().st_mtime if p.exists() else None


@st.cache_resource(show_spinner=False)
def _load_gate(mtime: float) -> pipeline.Snapshot | None:
    return pipeline.load()


@st.cache_resource(show_spinner=False)
def _load_scan(mtime: float) -> scan.ScanResult | None:
    return scan.load()


def gate_snapshot() -> pipeline.Snapshot:
    """Latest macro gate snapshot; builds one on first launch or when the saved one is incompatible."""
    m = _mtime(config.SNAPSHOT_FILE)
    snap = _load_gate(m) if m is not None else None
    if snap is None:
        with st.spinner("No usable gate snapshot. Downloading data and computing scores (a few minutes)…"):
            pipeline.save(pipeline.build(refresh=False))
        snap = _load_gate(_mtime(config.SNAPSHOT_FILE))
    return snap


def scan_result() -> scan.ScanResult | None:
    m = _mtime(config.SCAN_FILE)
    return _load_scan(m) if m is not None else None


@st.cache_resource(show_spinner=False)
def _load_analysis(mtime: float):
    from analyst import run as analyst_run
    return analyst_run.load()


def analysis_run():
    m = _mtime(config.ANALYSIS_FILE)
    return _load_analysis(m) if m is not None else None
