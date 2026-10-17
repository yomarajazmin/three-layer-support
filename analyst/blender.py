"""Score blender: 60% quant composite + 40% Claude fundamental score, then re-rank.

Rank changes of 3+ positions are flagged: these are where the quant model and
the AI analyst disagree, which is where the most interesting information lives.
"""

from __future__ import annotations

import pandas as pd

import config
from analyst.analyzer import AnalysisRecord


def blend(candidates: pd.DataFrame, analyses: dict[str, AnalysisRecord]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (blended, failed).

    `blended` holds the successfully analyzed tickers. `quant_rank` and
    `blended_rank` are both computed within that set, so `rank_change`
    (positive = moved up) compares like with like.
    `failed` lists tickers whose analysis errored; they are not re-ranked.
    """
    rows, failed = [], []
    for ticker, cand in candidates.iterrows():
        rec = analyses.get(ticker)
        if rec is None or rec.scores is None:
            failed.append({"ticker": ticker, "quant_score": cand["composite"],
                           "error": rec.error if rec else "not analyzed"})
            continue
        s = rec.scores
        rows.append({
            "ticker": ticker,
            "scanner_rank": int(cand["rank"]),
            "quant_score": float(cand["composite"]),
            "fundamental_score": rec.fundamental_score,
            "earnings_quality": s.earnings_quality.score,
            "growth_trajectory": s.growth_trajectory.score,
            "balance_sheet_health": s.balance_sheet_health.score,
            "margin_trends": s.margin_trends.score,
            "red_flags": s.red_flags.score,
            "quarter_end": rec.quarter_end,
            "cached": rec.cached,
        })

    cols = ["ticker", "scanner_rank", "quant_rank", "blended_rank", "rank_change", "flag", "quant_score",
            "fundamental_score", "blended_score", "earnings_quality", "growth_trajectory",
            "balance_sheet_health", "margin_trends", "red_flags", "quarter_end", "cached"]
    failed_df = pd.DataFrame(failed, columns=["ticker", "quant_score", "error"])
    if not rows:
        return pd.DataFrame(columns=cols).set_index("ticker"), failed_df

    df = pd.DataFrame(rows).set_index("ticker")
    df["blended_score"] = (config.BLEND_QUANT_WEIGHT * df["quant_score"]
                           + config.BLEND_FUNDAMENTAL_WEIGHT * df["fundamental_score"])
    # Ties broken by the quant score, then ticker, so ranks are deterministic.
    df["quant_rank"] = _rank(df, "quant_score")
    df["blended_rank"] = _rank(df, "blended_score", tiebreak="quant_score")
    df["rank_change"] = df["quant_rank"] - df["blended_rank"]          # + = upgraded by the analyst
    df["flag"] = "none"
    df.loc[df["rank_change"] >= config.RANK_FLAG_THRESHOLD, "flag"] = "upgrade"
    df.loc[df["rank_change"] <= -config.RANK_FLAG_THRESHOLD, "flag"] = "downgrade"
    return df.sort_values("blended_rank")[cols[1:]], failed_df


def _rank(df: pd.DataFrame, col: str, tiebreak: str | None = None) -> pd.Series:
    keys = [col] + ([tiebreak] if tiebreak else [])
    order = df.reset_index().sort_values(keys + ["ticker"], ascending=[False] * len(keys) + [True])["ticker"]
    return pd.Series(range(1, len(order) + 1), index=order.values).reindex(df.index)
