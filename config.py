"""Central configuration: tickers, lookbacks, score mappings, weights and deployment zones."""

# --- Tickers (Yahoo Finance symbols) ---
VIX = "^VIX"        # CBOE 30-day implied vol ("front-month" VIX)
VIX3M = "^VIX3M"    # CBOE 3-month implied vol
HYG = "HYG"         # iShares iBoxx High Yield Corporate Bond ETF
TLT = "TLT"         # iShares 20+ Year Treasury Bond ETF
SPY = "SPY"         # benchmark for the backtest
MACRO_TICKERS = [VIX, VIX3M, HYG, TLT, SPY]

# How much history to download. The crowding value proxy needs a 3-year
# lookback; add the 63-day correlation warm-up and a 2-year backtest window
# and ~6 years is required.
HISTORY_PERIOD = "6y"
TRADING_DAYS_1Y = 252
# Today's daily bar is treated as incomplete (and dropped) until this New York time.
# 16:00 close + time for the 16:15 VIX settlement and final volume prints.
MARKET_DATA_CUTOFF_ET = "16:30"
PRICE_MIN_COVERAGE = 0.98      # warn when fewer S&P 500 members than this have price data

# --- Signal parameters ---
# 1. VIX level: percentile vs trailing 1y, inverted (low VIX = high score)
VIX_LOW_BONUS_LEVEL = 15.0
VIX_LOW_BONUS = 5.0
VIX_HIGH_PENALTY_LEVEL = 30.0
VIX_HIGH_PENALTY = -10.0

# 2. VIX term structure: ratio VIX / VIX3M, linear map
TERM_RATIO_AT_100 = 0.85
TERM_RATIO_AT_0 = 1.15

# 3. Breadth: % of S&P 500 members above their 200-day SMA
BREADTH_SMA_DAYS = 200
BREADTH_PCT_AT_100 = 80.0
BREADTH_PCT_AT_0 = 30.0

# 4. Credit spreads: z-score of log(TLT/HYG) vs trailing 1y
CREDIT_Z_AT_100 = -2.0
CREDIT_Z_AT_0 = 2.0

# 5. Put/Call sentiment proxy: VIX 20-day rate of change (%)
PUTCALL_ROC_DAYS = 20
PUTCALL_ROC_AT_100 = -30.0
PUTCALL_ROC_AT_0 = 50.0

# 6. Factor crowding: rolling correlation of momentum vs value L/S factor returns
CROWDING_MOM_LOOKBACK = 252      # momentum = return from t-252 ...
CROWDING_MOM_SKIP = 21           # ... to t-21 (12-1 month)
CROWDING_VALUE_LOOKBACK = 756    # value proxy = long-term reversal: return from t-756 ...
CROWDING_VALUE_SKIP = 252        # ... to t-252 (3y excluding the last year); long losers, short winners
CROWDING_DECILE = 0.10           # long/short legs = top/bottom 10%, equal weight
CROWDING_MIN_STOCKS = 100        # minimum ranked stocks required to form baskets
CROWDING_CORR_WINDOW = 63
CROWDING_CORR_AT_100 = 0.3
CROWDING_CORR_AT_0 = -0.8

# --- Composite ---
SIGNAL_WEIGHTS = {
    "vix_level": 0.25,
    "vix_term_structure": 0.20,
    "breadth": 0.20,
    "credit_spreads": 0.15,
    "put_call": 0.10,
    "crowding": 0.10,
}

# Deployment zones, checked top-down: first zone whose `min_score` <= score wins.
ZONES = [
    {"name": "FULL DEPLOY", "min_score": 70.0, "sizing": 1.00,
     "allow_new_longs": True, "scanner_enabled": True,
     "guidance": "Full position sizing."},
    {"name": "REDUCED", "min_score": 40.0, "sizing": 0.60,
     "allow_new_longs": True, "scanner_enabled": True,
     "guidance": "60% sizing; higher bar for new positions."},
    {"name": "DEFENSIVE", "min_score": 0.0, "sizing": 0.25,
     "allow_new_longs": False, "scanner_enabled": False,
     "guidance": "25% sizing; no new longs; scanner disabled."},
]

# Scanner output filter per zone (None = scanner disabled, return an empty list).
SCANNER_MIN_SCORE = {"FULL DEPLOY": 0.0, "REDUCED": 75.0, "DEFENSIVE": None}

# --- Scanner ---
SCANNER_LOOKBACK_DAYS = 365      # calendar days sliced from the shared download
SCANNER_MIN_HISTORY = 200        # days of closes a stock needs to be scanned
SCAN_EMA_FAST = 10
SCAN_EMA_SLOW = 50
SCAN_CROSS_LOOKBACK = 5          # crossover must have happened within the last N trading days
SCAN_RET_3M_DAYS = 63
SCAN_VOL_SHORT = 5
SCAN_VOL_LONG = 20
SCAN_VOL_RATIO_AT_0 = 0.7
SCAN_VOL_RATIO_AT_100 = 2.0
SCAN_RS_DAYS = 20
SCAN_HIGH_DAYS = 252
SCAN_HIGH_RATIO_AT_100 = 0.95
SCAN_HIGH_RATIO_AT_0 = 0.70
SCAN_SHORT_MISSING_SCORE = 50.0
SCAN_SHORT_WORKERS = 8           # parallel yfinance .info requests
SCAN_SHORT_MIN_COVERAGE = 0.90  # warn when fewer tickers than this have short interest data
SCAN_FILE = "output/scan.pkl"

# --- Claude analyst layer ---
ANALYST_MODEL = "claude-opus-5-5"
ANALYST_EFFORT = "high"
ANALYST_MAX_TOKENS = 16000
ANALYST_TOP_N = 10               # candidates (by quant rank) sent to Claude per run
ANALYST_WORKERS = 4              # parallel Claude calls
ANALYST_QUARTERS = 4
ANALYST_DB = "output/analyst_cache.sqlite"
ANALYSIS_FILE = "output/analysis.pkl"
BLEND_QUANT_WEIGHT = 0.60
BLEND_FUNDAMENTAL_WEIGHT = 0.40
RANK_FLAG_THRESHOLD = 3          # |rank change| >= this is flagged upgrade / downgrade

# --- Backtest ---
BACKTEST_DAYS = 504  # ~2 years of trading days

# --- Paths ---
CACHE_DIR = ".cache"
OUTPUT_DIR = "output"
SNAPSHOT_FILE = "output/snapshot.pkl"
