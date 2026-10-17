import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Synthetic market + faked short interest, fundamentals and Claude, inside a temp cwd
    (output/ and .cache/ are relative paths, so the real ones are never touched)."""
    import pipeline
    from analyst import analyzer
    from scanner import data as scan_data
    from synthetic import FakeClaude, fake_fundamentals, fake_short, synthetic_market

    monkeypatch.chdir(tmp_path)
    m = synthetic_market()
    monkeypatch.setattr(pipeline, "load_market_data", lambda refresh=False: m)
    monkeypatch.setattr(scan_data, "load_market_data", lambda refresh=False: m)
    monkeypatch.setattr(scan_data, "load_short_interest", fake_short)
    monkeypatch.setattr(analyzer, "get_fundamentals", fake_fundamentals)
    claude = FakeClaude()
    monkeypatch.setattr(analyzer, "make_client", lambda: claude)
    return SimpleNamespace(market=m, claude=claude, tmp=tmp_path)
