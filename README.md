# Three-Layer Market Deployment System

Answers three questions in order, each layer gated by the one before it:

1. **Macro gate**: *Should I be deploying capital right now, and how aggressively?*
2. **Stock scanner**: *If so, which S&P 500 stocks rank best on quant factors?*
3. **AI analyst**: *Of those, which hold up on fundamentals, and where do the quant model and Claude disagree?*

> Personal research tool. Not investment advice. The AI layer is nondeterministic and can be wrong.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt     # Windows (macOS/Linux: .venv/bin/python)
cp .env.example .env    # then set ANTHROPIC_API_KEY (needed only for layer 3)
```

## Running

| Command | What it does |
|---|---|
| `python run_macro_gate.py` | Downloads market data, scores the 6 macro signals, runs the 2-year backtest, saves `output/snapshot.pkl`. `--use-cache` reuses today's download. |
| `python run_scanner.py` | Scans the S&P 500 if the gate allows it, saves `output/scan.pkl` + `scan_candidates.csv`. `--refresh` forces new data, `--top N` rows to print. |
| `python run_analysis.py --scan-and-analyze` | Full end-to-end: gate, then scanner, then Claude on the top 10 candidates, then blend. Saves `output/analysis.pkl` + `analysis_blended.csv`. Without the flag it analyzes the last saved scan. `--top N`, `--force` (ignore the analysis cache), `--refresh`. |
| `streamlit run app.py` | Dashboard: **Macro Gate**, **Stock Scanner**, **AI Analyst** pages. |
| `python -m pytest` | Test suite (no network, no paid API calls). |

## Layer 1: Macro gate (`signals/`, `pipeline.py`, `backtest/`)

Six signals, each scored 0–100, then blended by weight (`config.SIGNAL_WEIGHTS`):

| Signal | Weight | Input | Score map |
|---|---|---|---|
| VIX level | 0.25 | VIX percentile vs trailing 1y | 100 − percentile; +5 if VIX < 15, −10 if > 30 |
| Term structure | 0.20 | VIX / VIX3M | 0.85 → 100, 1.15 → 0 |
| Breadth | 0.20 | % of S&P 500 above 200-day SMA | 80% → 100, 30% → 0 |
| Credit spreads | 0.15 | z-score of log(TLT/HYG), 1y | z −2 → 100, +2 → 0 |
| Put/Call proxy | 0.10 | VIX 20-day rate of change | −30% → 100, +50% → 0 |
| Factor crowding | 0.10 | 63-day corr of momentum vs value L/S returns | +0.3 → 100, −0.8 → 0 |

Zones (`config.ZONES`): **FULL DEPLOY** ≥ 70 (100% sizing), **REDUCED** 40–69 (60%, higher bar for new positions), **DEFENSIVE** < 40 (25%, no new longs, scanner disabled).

The backtest recomputes the score daily over 2 years. Each day's zone comes from the **previous** close's score (no look-ahead), and SPY returns are compared by zone.

## Layer 2: Stock scanner (`scanner/`)

Five factors, 0–100 each, equal-weighted:

1. **Momentum crossover**: 10/50-day EMA cross in the last 5 days. Stocks that crossed are ranked on EMA gap + 3-month return; everyone else scores 0.
2. **Volume surge**: 5d / 20d average volume, 0.7 → 0, 2.0 → 100.
3. **Relative strength**: percentile of (20-day return − SPY 20-day return).
4. **52-week high proximity**: close / 52-week high close, ≥ 0.95 → 100, 0.70 → 0.
5. **Short interest decline**: percentile of the drop in shares short vs the prior month. Missing data scores a neutral 50.

The gate decides what is surfaced: FULL DEPLOY = all stocks ranked, REDUCED = composite > 75 only, DEFENSIVE = scanner disabled and nothing downloaded.

## Layer 3: AI analyst (`analyst/`)

- `fundamentals.py`: last 4 quarters from yfinance (revenue, net income, CFO, FCF, margins, D/E, ROE, CFO/NI, AR vs revenue growth YoY). Missing items are sent as missing, never estimated.
- `analyzer.py`: `claude-opus-5-5` (effort high, structured JSON output, server-side refusal fallback) scores Earnings Quality, Growth, Balance Sheet, Margins and Red Flags 1–10 (10 = best / no red flags). Fundamental score = mean × 10.
- `blender.py`: 60% quant + 40% fundamental, re-ranked. Moves of ≥ 3 places are flagged as AI upgrades or downgrades.
- **Cache**: SQLite `output/analyst_cache.sqlite`, keyed by (ticker, quarter end, version). The same quarter is free. `version` hashes the model, effort, prompts, schema and payload format, so changing any of them triggers fresh analyses. Bump `fundamentals.PAYLOAD_VERSION` when you change what data is sent.
- **Cost**: up to 10 Opus calls per run on days with candidates, typically about $1. Cached quarters cost nothing.

## Data and caching

- **One shared download per day** (`data/fetch.py`): 6 years of OHLCV for the S&P 500 plus VIX, VIX3M, HYG, TLT and SPY, cached in `.cache/`. The gate uses all of it; the scanner slices the last year, so both layers always see the same prices.
- **No partial bars** (`data/market_hours.py`): on a trading day, today's bar is dropped until 16:30 ET. A same-day cache made before then is re-downloaded after it.
- **Stale data** is flagged on the dashboard. The CLIs rebuild the gate when its data predates the latest completed session (holidays aren't modeled).
- **Data-quality warnings** are shown on every page and printed by the CLIs: missing price data, low short-interest coverage (e.g. Yahoo rate limiting), failed analyses.
- **Snapshots** (`storage.py`) carry a schema fingerprint. A file written by an incompatible code version is ignored and rebuilt instead of crashing.

## Known limitations

- **Survivorship bias**: breadth, crowding and the backtest use *today's* S&P 500 members for all of history. Current readings are unaffected.
- **Proxies**: credit spread = TLT vs HYG (TLT also moves with long rates); put/call = VIX momentum; value = 3-year price reversal (no point-in-time fundamentals).
- **Crowding** rarely nears its +0.3 "normal" anchor (median 63-day correlation about −0.46), so it usually scores low.
- **REDUCED mode** rarely surfaces stocks: without a fresh EMA cross a stock can score at most 80.
- **Undeployed capital** earns 0% in the backtest's strategy comparison.

## Layout

```
app.py                 Streamlit entry (3 pages in views/)
config.py              every threshold, weight, path and model setting
pipeline.py            layer 1 orchestration + snapshot
storage.py             versioned snapshot files
data/                  shared market data download, market-hours guard
signals/               six macro signals + composite
backtest/              no-look-ahead zone backtest
scanner/               factors, gating, scanner data
analyst/               fundamentals, Claude analyzer + cache, blender, run
views/                 dashboard pages + theme
tests/                 unit, robustness and end-to-end tests
run_*.py               CLI entry points
```
