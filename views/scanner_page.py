"""Page 2: Gated multi-factor stock scanner."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from scanner import scan
from views.state import gate_snapshot, scan_result
from views.theme import SERIES_1, SERIES_2, TEXT, TEXT_MUTED, ZONE_COLOR, layout, show, zone_badge

SERIES_3 = "#199e70"   # dark-mode categorical slot 3 (aqua)
FACTOR_COLS = {f"{k}_score": name for k, name in scan.FACTORS.items()}


def run_scan(refresh: bool) -> None:
    gate = gate_snapshot()
    msg = "Scanning the S&P 500 (first run downloads OHLCV + ~500 short interest records)…"
    with st.spinner(msg):
        scan.save(scan.run(gate, refresh=refresh))
    st.rerun()


gate = gate_snapshot()
zone = gate.composite.zone
result = scan_result()

with st.sidebar:
    st.header("Scanner")
    force = st.checkbox("Force fresh data download", value=False)
    if st.button("Run scan", width="stretch"):
        run_scan(refresh=force)
    st.markdown("<div class='muted'>CLI: <code>python run_scanner.py</code><br>"
                "Factors (equal weight): momentum crossover, volume surge, relative strength vs SPY, "
                "52-week high proximity, short interest decline.</div>", unsafe_allow_html=True)
    st.header("Gate rules")
    for z in config.ZONES:
        rule = config.SCANNER_MIN_SCORE[z["name"]]
        text = "disabled" if rule is None else ("all ranked" if rule == 0 else f"composite > {rule:.0f}")
        st.markdown(f"<span style='color:{ZONE_COLOR[z['name']]}'>●</span> **{z['name']}**: {text}",
                    unsafe_allow_html=True)
    st.caption("Not investment advice.")

st.title("Stock Scanner")
st.caption("Multi-factor ranking of the S&P 500, active only when the macro gate allows it.")

# First visit: a disabled scan needs no data, so create it immediately; otherwise ask to run.
if result is None:
    if not zone["scanner_enabled"]:
        scan.save(scan.run(gate))
        st.rerun()
    st.info(f"Macro gate is {zone['name']}. No scan yet. Click **Run scan** in the sidebar.")
    st.stop()

stale = (result.gate_generated_at != gate.generated_at) or (result.gate_zone["name"] != zone["name"])
if stale:
    st.warning(f"This scan used the gate snapshot from {result.gate_generated_at:%Y-%m-%d %H:%M} "
               f"({result.gate_zone['name']}); the current gate is {zone['name']} "
               f"({gate.generated_at:%Y-%m-%d %H:%M}). Run the scan again to apply it.")

if result.is_stale():
    st.warning(f"Scan prices end {result.as_of:%Y-%m-%d}, before the latest completed session. "
               "Run the scan again with fresh data.")
for w in result.warnings:
    st.warning(w)

rz = result.gate_zone
color = ZONE_COLOR[rz["name"]]
if result.status == "disabled":
    state = "<b>DISABLED</b>: macro gate is DEFENSIVE, so no candidates are returned."
elif result.status == "filtered":
    state = f"<b>ACTIVE, filtered</b>: only stocks with composite score above {result.min_score:.0f}."
else:
    state = "<b>ACTIVE</b>: all stocks ranked."
st.markdown(f"<div class='banner' style='border-color:{color}'>{zone_badge(rz['name'])}"
            f"<div style='margin-top:0.6rem'>Gate score {result.gate_score:.1f} · Scanner {state}</div>"
            f"<div class='muted'>Scan run {result.run_at:%Y-%m-%d %H:%M}"
            + (f" · price data as of {result.as_of:%Y-%m-%d}" if result.as_of is not None else "")
            + "</div></div>", unsafe_allow_html=True)

if result.status == "disabled":
    st.dataframe(pd.DataFrame(columns=["rank", "ticker", "composite"]), width="stretch", hide_index=True)
    st.stop()

ranked, cands = result.ranked, result.candidates
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Stocks scanned", f"{result.universe_size}")
m2.metric("Candidates", f"{len(cands)}")
m3.metric("Fresh EMA crossovers", f"{int(ranked['crossed'].sum())}")
m4.metric("Short interest coverage", f"{result.short_coverage}/{result.universe_size}")
m5.metric("Excluded (short history)", f"{len(result.excluded)}")

# Filters (FULL DEPLOY shows everything by default; REDUCED is already floored at 75 by the gate)
f1, f2, f3 = st.columns([1, 1, 1])
with f1:
    floor = float(result.min_score or 0.0)
    min_score = st.slider("Minimum composite", min_value=floor, max_value=100.0, value=floor, step=1.0,
                          disabled=len(cands) == 0)
with f2:
    top_n = st.number_input("Show top N", min_value=1, max_value=max(1, len(cands)),
                            value=min(25, max(1, len(cands))), step=5)
with f3:
    crossers_only = st.checkbox("Only fresh EMA crossovers")

view = cands[cands["composite"] >= min_score]
if crossers_only:
    view = view[view["crossed"]]
view = view.head(int(top_n))

if view.empty:
    st.info("No stocks pass the current filters." if not cands.empty else
            f"No stocks scored above {result.min_score:.0f} today "
            f"(highest composite in the universe: {ranked['composite'].max():.1f}).")
    st.stop()

table = view.reset_index()[["rank", "ticker", "price", "composite", *FACTOR_COLS, "crossed", "ema_gap_pct",
                            "ret_3m_pct", "vol_ratio", "rs_spread_pct", "high_ratio", "si_change_pct"]]
score_cfg = {c: st.column_config.ProgressColumn(n, min_value=0, max_value=100, format="%.0f")
             for c, n in FACTOR_COLS.items()}
st.dataframe(
    table, width="stretch", hide_index=True, height=min(38 * (len(table) + 1), 720),
    column_config={
        "rank": st.column_config.NumberColumn("#", format="%d"),
        "ticker": "Ticker",
        "price": st.column_config.NumberColumn("Price", format="%.2f"),
        "composite": st.column_config.ProgressColumn("Composite", min_value=0, max_value=100, format="%.1f"),
        **score_cfg,
        "crossed": st.column_config.CheckboxColumn("Fresh cross"),
        "ema_gap_pct": st.column_config.NumberColumn("EMA gap %", format="%.2f"),
        "ret_3m_pct": st.column_config.NumberColumn("3M ret %", format="%.1f"),
        "vol_ratio": st.column_config.NumberColumn("Vol 5d/20d", format="%.2f"),
        "rs_spread_pct": st.column_config.NumberColumn("RS vs SPY %", format="%+.1f"),
        "high_ratio": st.column_config.NumberColumn("Px / 52W high", format="%.3f"),
        "si_change_pct": st.column_config.NumberColumn("SI change %", format="%+.1f"),
    },
)
st.download_button("Download candidates (CSV)", cands.to_csv().encode(), "scan_candidates.csv", "text/csv")

# Detail for one ticker
st.subheader("Candidate detail")
ticker = st.selectbox("Ticker", list(view.index))
row = ranked.loc[ticker]
d1, d2 = st.columns([2, 1])
with d1:
    closes = result.closes
    if ticker in closes.columns:
        px = closes[ticker].dropna()
        fast = px.ewm(span=config.SCAN_EMA_FAST, adjust=False).mean()
        slow = px.ewm(span=config.SCAN_EMA_SLOW, adjust=False).mean()
        fig = go.Figure()
        for s, name, c, w in ((px, "Close", SERIES_1, 2), (fast, "EMA 10", SERIES_2, 1.5), (slow, "EMA 50", SERIES_3, 1.5)):
            fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=name,
                                     line=dict(color=c, width=w), hovertemplate=f"{name} %{{y:.2f}}<extra></extra>"))
        show(layout(fig, f"{ticker}: close with 10/50-day EMAs (1 year)", height=360, legend=True))
    else:
        st.info(f"No price history stored for {ticker}.")
with d2:
    scores = pd.Series({n: row[c] for c, n in FACTOR_COLS.items()})[::-1]
    fig = go.Figure(go.Bar(x=scores.values, y=scores.index, orientation="h", marker_color=SERIES_1,
                           text=[f"{v:.0f}" for v in scores.values], textposition="outside",
                           textfont=dict(color=TEXT), hovertemplate="%{y}: %{x:.1f}<extra></extra>"))
    layout(fig, f"{ticker} factor scores (composite {row['composite']:.1f})", height=360)
    fig.update_xaxes(range=[0, 110], showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    fig.update_layout(hovermode="closest", bargap=0.35, margin=dict(l=170, r=10, t=40, b=10))
    fig.update_yaxes(tickfont=dict(color=TEXT))
    show(fig)
    si_date = row["si_date"]
    st.markdown(f"<div class='muted'>Short interest report: "
                f"{'n/a' if pd.isna(si_date) else f'{si_date:%Y-%m-%d}'} (vs prior month). "
                f"Missing short interest scores a neutral 50.</div>", unsafe_allow_html=True)
