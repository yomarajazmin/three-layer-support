"""Quarterly fundamentals from yfinance for the analyst layer.

Per quarter (last 4): revenue, net income, operating cash flow, free cash flow,
gross margin, operating margin, debt/equity, ROE (quarterly, not annualized),
CFO/NI. Plus AR growth vs revenue growth: year-over-year (latest quarter vs the
same quarter a year earlier) when 5 quarters exist, otherwise across the
available quarters. Missing line items are reported as None, never estimated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd
import yfinance as yf

import config

# Bump when the payload sent to Claude changes meaning (new fields, formula fixes),
# so cached analyses built from the old payload are not reused.
PAYLOAD_VERSION = 1


@dataclass
class Fundamentals:
    ticker: str
    quarter_end: str                     # latest quarter end, ISO date (cache key)
    quarters: list[dict]                 # newest first
    derived: dict
    missing: list[str] = field(default_factory=list)
    profile: dict = field(default_factory=dict)   # name / sector / industry

    def to_payload(self) -> dict:
        return {"ticker": self.ticker, **self.profile, "latest_quarter_end": self.quarter_end,
                "quarters_newest_first": self.quarters, "derived_metrics": self.derived,
                "unavailable_line_items": self.missing}


def _row(df: pd.DataFrame, *names: str) -> pd.Series | None:
    for n in names:
        if df is not None and n in df.index:
            return df.loc[n]
    return None


def _num(x) -> float | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) or math.isinf(x) else x


def _ratio(a, b) -> float | None:
    a, b = _num(a), _num(b)
    return None if a is None or b is None or b == 0 else a / b


def _growth(new, old) -> float | None:
    new, old = _num(new), _num(old)
    return None if new is None or old is None or old == 0 else (new - old) / abs(old)


def get_fundamentals(ticker: str, n_quarters: int = config.ANALYST_QUARTERS) -> Fundamentals:
    tk = yf.Ticker(ticker)
    inc, cf, bs = tk.quarterly_income_stmt, tk.quarterly_cashflow, tk.quarterly_balance_sheet
    if inc is None or inc.empty:
        raise RuntimeError(f"No quarterly income statement for {ticker}")

    rows = {
        "revenue": _row(inc, "Total Revenue", "Operating Revenue"),
        "net_income": _row(inc, "Net Income", "Net Income Common Stockholders"),
        "gross_profit": _row(inc, "Gross Profit"),
        "operating_income": _row(inc, "Operating Income"),
        "operating_cash_flow": _row(cf, "Operating Cash Flow"),
        "free_cash_flow": _row(cf, "Free Cash Flow"),
        "capex": _row(cf, "Capital Expenditure"),
        "total_debt": _row(bs, "Total Debt"),
        "equity": _row(bs, "Stockholders Equity", "Common Stock Equity"),
        "accounts_receivable": _row(bs, "Accounts Receivable", "Receivables"),
    }
    missing = [k for k, v in rows.items() if v is None]

    def val(key, date):
        s = rows[key]
        return _num(s.get(date)) if s is not None else None

    dates = sorted(inc.columns, reverse=True)
    dates = [d for d in dates if val("revenue", d) is not None or val("net_income", d) is not None]
    if not dates:
        raise RuntimeError(f"No usable quarterly revenue / net income for {ticker}")
    recent = dates[:n_quarters]

    quarters = []
    for d in recent:
        rev, ni, cfo = val("revenue", d), val("net_income", d), val("operating_cash_flow", d)
        fcf = val("free_cash_flow", d)
        if fcf is None and cfo is not None and val("capex", d) is not None:
            fcf = cfo + val("capex", d)   # capex is reported as a negative number
        quarters.append({
            "quarter_end": pd.Timestamp(d).date().isoformat(),
            "revenue": rev,
            "net_income": ni,
            "operating_cash_flow": cfo,
            "free_cash_flow": fcf,
            "gross_margin": _ratio(val("gross_profit", d), rev),
            "operating_margin": _ratio(val("operating_income", d), rev),
            "debt_to_equity": _ratio(val("total_debt", d), val("equity", d)),
            "roe_quarterly": _ratio(ni, val("equity", d)),
            "cfo_to_net_income": _ratio(cfo, ni),
            "accounts_receivable": val("accounts_receivable", d),
        })

    # AR growth vs revenue growth: YoY if a year-ago quarter exists, else oldest -> latest of `recent`.
    latest = dates[0]
    base = dates[4] if len(dates) > 4 else recent[-1]
    basis = "yoy" if len(dates) > 4 else f"{len(recent) - 1}q_span"
    ar_g = _growth(val("accounts_receivable", latest), val("accounts_receivable", base))
    rev_g = _growth(val("revenue", latest), val("revenue", base))

    full_year = len(quarters) == n_quarters == 4
    ttm = {k: sum(q[k] for q in quarters) if full_year and all(q[k] is not None for q in quarters) else None
           for k in ("revenue", "net_income", "operating_cash_flow", "free_cash_flow")}
    derived = {
        "ttm_revenue": ttm["revenue"],
        "ttm_net_income": ttm["net_income"],
        "ttm_operating_cash_flow": ttm["operating_cash_flow"],
        "ttm_free_cash_flow": ttm["free_cash_flow"],
        "ttm_cfo_to_net_income": _ratio(ttm["operating_cash_flow"], ttm["net_income"]),
        "ttm_roe": _ratio(ttm["net_income"], val("equity", latest)),
        "ar_growth": ar_g,
        "revenue_growth": rev_g,
        "ar_growth_minus_revenue_growth": None if ar_g is None or rev_g is None else ar_g - rev_g,
        "growth_basis": basis,
        "growth_from_quarter": pd.Timestamp(base).date().isoformat(),
        "growth_to_quarter": pd.Timestamp(latest).date().isoformat(),
        "ttm_complete": all(v is not None for v in ttm.values()),
    }
    try:
        info = tk.info or {}
    except Exception:
        info = {}
    profile = {"company_name": info.get("longName") or info.get("shortName"),
               "sector": info.get("sector"), "industry": info.get("industry")}
    return Fundamentals(ticker, pd.Timestamp(latest).date().isoformat(), quarters, derived, missing, profile)
