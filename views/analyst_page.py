"""Page 3: Claude analyst layer - blended ranking, rank deltas, expandable analysis."""

from __future__ import annotations

import html
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

import config
from analyst import run as analyst_run
from analyst.analyzer import DIMENSIONS
from scanner import scan
from views.state import analysis_run, gate_snapshot, scan_result
from views.theme import SURFACE, TEXT, TEXT_MUTED, ZONE_COLOR, layout, show, zone_badge

UP, DOWN, NEUTRAL = "#0ca30c", "#d03b3b", "#6b7180"
SUB_COLS = {"earnings_quality": "EQ", "growth_trajectory": "Growth", "balance_sheet_health": "BS",
            "margin_trends": "Margins", "red_flags": "Flags"}

load_dotenv()
has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))

with st.sidebar:
    st.header("AI Analyst")
    top_n = st.number_input("Candidates to analyze", min_value=1, max_value=50, value=config.ANALYST_TOP_N)
    force = st.checkbox("Ignore cache (re-run Claude)", value=False)
    run_clicked = st.button("Analyze last scan", width="stretch",
                            help=f"Calls {config.ANALYST_MODEL} for candidates whose latest quarter isn't cached.")
    full_clicked = st.button("Scan + analyze (end-to-end)", width="stretch")
    if not has_key:
        st.warning("ANTHROPIC_API_KEY not found in .env. Only cached analyses can be shown.")
    st.markdown(f"<div class='muted'>Model: <code>{config.ANALYST_MODEL}</code> · effort {config.ANALYST_EFFORT}<br>"
                f"Blend: {config.BLEND_QUANT_WEIGHT:.0%} quant + {config.BLEND_FUNDAMENTAL_WEIGHT:.0%} fundamental<br>"
                f"Flag: rank change ≥ {config.RANK_FLAG_THRESHOLD}<br>"
                "Cache: SQLite by (ticker, quarter end); same quarter = no new call.<br>"
                "CLI: <code>python run_analysis.py --scan-and-analyze</code></div>", unsafe_allow_html=True)
    st.caption("AI output is nondeterministic and can be wrong. Not investment advice.")

if run_clicked or full_clicked:
    with st.spinner("Running analysis (Claude calls can take a minute each)…"):
        res = scan_result()
        if full_clicked or res is None:
            res = scan.run(gate_snapshot())
            scan.save(res)
        analyst_run.save(analyst_run.run(res, top_n=int(top_n), force=force))
    st.rerun()

st.title("AI Analyst")
st.caption("Claude scores the top scanner candidates on fundamental quality; the blend re-ranks them. "
           "The flagged rank changes are where the quant model and the analyst disagree.")

run = analysis_run()
current_scan = scan_result()
if run is None:
    st.info("No analysis yet. Use **Analyze last scan** or **Scan + analyze** in the sidebar.")
    st.stop()

if current_scan is not None and current_scan.run_at != run.scan_run_at:
    st.warning(f"This analysis used the scan from {run.scan_run_at:%Y-%m-%d %H:%M}; a newer scan exists "
               f"({current_scan.run_at:%Y-%m-%d %H:%M}). Re-run the analysis to apply it.")

zc = ZONE_COLOR[run.gate_zone["name"]]
st.markdown(
    f"<div class='banner' style='border-color:{zc}'>{zone_badge(run.gate_zone['name'])}"
    f"<div style='margin-top:0.6rem'>Gate score {run.gate_score:.1f} · scanner {run.scan_status} · "
    f"analyzed top {run.top_n} with <code>{run.model}</code></div>"
    f"<div class='muted'>Run {run.run_at:%Y-%m-%d %H:%M} · Claude calls {run.claude_calls} · cache hits "
    f"{run.cache_hits} · tokens in/out {run.input_tokens:,}/{run.output_tokens:,}</div></div>",
    unsafe_allow_html=True)

for w in run.warnings:
    st.warning(w)
if run.note:
    st.info(run.note)
if not run.failed.empty:
    with st.expander(f"{len(run.failed)} candidate(s) could not be analyzed"):
        st.dataframe(run.failed, hide_index=True, width="stretch")
b = run.blended
if b.empty:
    st.stop()

ups, downs = b[b["flag"] == "upgrade"], b[b["flag"] == "downgrade"]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Analyzed", len(b))
m2.metric("Upgraded ≥3", len(ups))
m3.metric("Downgraded ≥3", len(downs))
m4.metric("Avg fundamental score", f"{b['fundamental_score'].mean():.0f}")


# ---------------------------------------------------------------- blended table (glow rows)
def delta_cell(d: int) -> str:
    if d > 0:
        return f"<span class='delta-up'>▲ +{d}</span>"
    if d < 0:
        return f"<span class='delta-down'>▼ {d}</span>"
    return "<span class='muted'>0</span>"


def flag_cell(flag: str) -> str:
    return {"upgrade": "<span class='delta-up'>AI upgrade</span>",
            "downgrade": "<span class='delta-down'>AI downgrade</span>"}.get(flag, "<span class='muted'>—</span>")


head = ["Blended #", "Ticker", "Flag", "Δ rank", "Quant #", "Quant", "Fundamental", "Blended",
        *SUB_COLS.values(), "Quarter"]
rows = []
for t, r in b.iterrows():
    cls = {"upgrade": "up", "downgrade": "down"}.get(r["flag"], "")
    cells = [int(r["blended_rank"]), f"<b>{html.escape(t)}</b>", flag_cell(r["flag"]), delta_cell(int(r["rank_change"])),
             int(r["quant_rank"]), f"{r['quant_score']:.1f}", f"{r['fundamental_score']:.0f}",
             f"<b>{r['blended_score']:.1f}</b>", *[int(r[c]) for c in SUB_COLS],
             f"{r['quarter_end']}{' · cached' if r['cached'] else ''}"]
    rows.append(f"<tr class='{cls}'>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
st.subheader("Blended ranking")
st.markdown("<table class='blend'><thead><tr>" + "".join(f"<th>{h}</th>" for h in head)
            + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>", unsafe_allow_html=True)
st.caption("Δ rank = quant rank − blended rank among the analyzed set (+ = the analyst moved it up). "
           "Sub-scores are 1–10 (Flags: 10 = no red flags). Fundamental = mean × 10.")

# ---------------------------------------------------------------- rank-change slope chart
fig = go.Figure()
for t, r in b.iterrows():
    color = UP if r["flag"] == "upgrade" else DOWN if r["flag"] == "downgrade" else NEUTRAL
    width = 3 if r["flag"] != "none" else 1.5
    fig.add_trace(go.Scatter(
        x=["Quant rank", "Blended rank"], y=[r["quant_rank"], r["blended_rank"]], mode="lines+markers+text",
        line=dict(color=color, width=width), marker=dict(size=9, color=color, line=dict(color=SURFACE, width=2)),
        text=[t, f"{t} ({int(r['rank_change']):+d})"], textposition=["middle left", "middle right"],
        textfont=dict(color=TEXT if r["flag"] != "none" else TEXT_MUTED, size=12), name=t,
        hovertemplate=f"{t}<br>%{{x}}: %{{y}}<extra></extra>"))
layout(fig, "Where the quant model and the analyst disagree", height=max(320, 34 * len(b) + 80))
fig.update_yaxes(autorange="reversed", dtick=1, title=None)
fig.update_xaxes(range=[-0.6, 1.6])
fig.update_layout(hovermode="closest")
show(fig)

# ---------------------------------------------------------------- expandable analysis
st.subheader("Analyst notes")
for t, r in b.iterrows():
    rec = run.records[t]
    s = rec.scores
    label = (f"#{int(r['blended_rank'])}  {t}: fundamental {r['fundamental_score']:.0f} · blended "
             f"{r['blended_score']:.1f} · Δ {int(r['rank_change']):+d}"
             + ("  ▲ AI upgrade" if r["flag"] == "upgrade" else "  ▼ AI downgrade" if r["flag"] == "downgrade" else ""))
    with st.expander(label, expanded=r["flag"] != "none"):
        prof = rec.fundamentals or {}
        st.markdown(f"<div class='muted'>{html.escape(str(prof.get('company_name') or t))} · "
                    f"{html.escape(str(prof.get('sector') or 'n/a'))} / {html.escape(str(prof.get('industry') or 'n/a'))} · "
                    f"latest quarter {rec.quarter_end} · {'cached' if rec.cached else 'new'} analysis "
                    f"({rec.model}, {rec.created_at})</div>", unsafe_allow_html=True)
        st.markdown(f"**Summary.** {s.summary}")
        cols = st.columns(len(DIMENSIONS))
        for col, (k, name) in zip(cols, DIMENSIONS.items()):
            d = getattr(s, k)
            col.metric(name, f"{d.score}/10")
            col.caption(d.rationale)
        if s.red_flag_items:
            st.markdown("**Red flags**\n" + "\n".join(f"- {item}" for item in s.red_flag_items))
        else:
            st.markdown("**Red flags:** none identified.")
        if prof.get("quarters_newest_first"):
            q = pd.DataFrame(prof["quarters_newest_first"]).set_index("quarter_end")
            money = ["revenue", "net_income", "operating_cash_flow", "free_cash_flow", "accounts_receivable"]
            for c in money:
                if c in q:
                    q[c] = q[c] / 1e6
            st.markdown("**Data sent to Claude** (money in millions; ratios as decimals)")
            st.dataframe(q.T.round(3), width="stretch")
            der = prof.get("derived_metrics", {})
            st.json(der, expanded=False)
