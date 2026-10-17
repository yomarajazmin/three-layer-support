import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def no_real_api_key(monkeypatch):
    """Tests never see the real ANTHROPIC_API_KEY: .env loading is disabled and the variable removed,
    so no test can make a paid Claude call even when a .env file exists."""
    import dotenv

    from analyst import analyzer
    noop = lambda *a, **k: False
    monkeypatch.setattr(dotenv, "load_dotenv", noop)
    monkeypatch.setattr(analyzer, "load_dotenv", noop)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


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
