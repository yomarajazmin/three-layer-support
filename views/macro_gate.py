"""Page 1: Macro deployment gate."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
import pipeline
from signals import breadth, credit_spreads, crowding, put_call, vix_level, vix_term_structure
from views.state import gate_snapshot
from views.theme import (BG, SERIES_1, SERIES_2, TEXT, TEXT_MUTED, ZONE_COLOR, layout, line_chart, show,
                         zone_badge)

DISPLAY_DAYS = config.BACKTEST_DAYS


# Per-signal raw-value chart settings: title, reference lines (value, label), y format.
RAW_CHART = {
    vix_level.KEY: ("VIX", [(config.VIX_LOW_BONUS_LEVEL, "15: +5 bonus"),
                            (config.VIX_HIGH_PENALTY_LEVEL, "30: −10 penalty")], ".2f"),
    vix_term_structure.KEY: ("VIX / VIX3M ratio", [(config.TERM_RATIO_AT_100, "0.85 → 100"),
                                                   (1.0, "1.00 (50)"),
                                                   (config.TERM_RATIO_AT_0, "1.15 → 0")], ".3f"),
    breadth.KEY: ("% of S&P 500 above 200-day SMA", [(config.BREADTH_PCT_AT_100, "80% → 100"),
                                                      (config.BREADTH_PCT_AT_0, "30% → 0")], ".1f"),
    credit_spreads.KEY: ("Spread proxy z-score, log(TLT/HYG)", [(config.CREDIT_Z_AT_100, "z −2 → 100"),
                                                                (0.0, "z 0"),
                                                                (config.CREDIT_Z_AT_0, "z +2 → 0")], "+.2f"),
    put_call.KEY: ("VIX 20-day rate of change (%)", [(config.PUTCALL_ROC_AT_100, "−30% → 100"),
                                                      (0.0, "0%"),
                                                      (config.PUTCALL_ROC_AT_0, "+50% → 0")], "+.1f"),
    crowding.KEY: ("63-day corr: momentum vs value L/S", [(config.CROWDING_CORR_AT_100, "+0.3 → 100"),
                                                          (0.0, "0"),
                                                          (config.CROWDING_CORR_AT_0, "−0.8 → 0")], "+.2f"),
}


def gauge(score: float) -> go.Figure:
    steps, upper = [], 100.0
    for zone in config.ZONES:
        steps.append({"range": [zone["min_score"], upper], "color": ZONE_COLOR[zone["name"]] + "40"})
        upper = zone["min_score"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number={"valueformat": ".0f", "font": {"size": 34, "color": TEXT}},
        gauge={
            "axis": {"range": [0, 100], "tickvals": [0, 40, 70, 100], "tickfont": {"size": 10, "color": TEXT_MUTED}},
            "bar": {"color": TEXT, "thickness": 0.28},
            "bgcolor": BG, "borderwidth": 0, "steps": steps,
        },
    ))
    fig.update_layout(height=135, margin=dict(l=30, r=30, t=14, b=0), paper_bgcolor=BG)
    return fig


def spy_zone_chart(daily: pd.DataFrame) -> go.Figure:
    """SPY close, line colored by the zone in effect that day (from the prior day's score)."""
    fig = go.Figure()
    for zone in config.ZONES:
        name = zone["name"]
        in_zone = daily["zone"] == name
        # Include the day before each run so colored segments connect without gaps.
        mask = in_zone | in_zone.shift(-1, fill_value=False)
        fig.add_trace(go.Scatter(
            x=daily.index, y=daily["spy"].where(mask), mode="lines", name=name, connectgaps=False,
            line=dict(color=ZONE_COLOR[name], width=2),
            customdata=daily[["score_prev"]].assign(zone=daily["zone"]).values,
            hovertemplate="SPY %{y:.2f}<br>Zone %{customdata[1]} (score %{customdata[0]:.1f})<extra></extra>",
        ))
    layout(fig, "SPY, colored by deployment zone in effect (prior day's score)", height=420, legend=True)
    fig.update_layout(hovermode="closest")
    return fig


# ---------------------------------------------------------------- page
snap = gate_snapshot()
comp = snap.composite
zone = comp.zone

with st.sidebar:
    st.header("Data")
    if st.button("Refresh data & recalculate", width="stretch"):
        with st.spinner("Downloading data and recalculating (a few minutes)…"):
            pipeline.save(pipeline.build(refresh=True))
        st.rerun()
    st.markdown(f"<div class='muted'>Snapshot: {snap.generated_at:%Y-%m-%d %H:%M}<br>"
                f"Data downloaded: {snap.data_downloaded_at:%Y-%m-%d %H:%M}<br>"
                "Source: Yahoo Finance (yfinance); S&P 500 members from Wikipedia.<br>"
                "CLI: <code>python run_macro_gate.py</code></div>", unsafe_allow_html=True)
    st.header("Zones")
    for z in config.ZONES:
        st.markdown(f"<span style='color:{ZONE_COLOR[z['name']]}'>●</span> **{z['name']}** "
                    f"≥ {z['min_score']:.0f}<br><span class='muted'>{z['guidance']}</span>",
                    unsafe_allow_html=True)
    st.header("Weights")
    for key, w in config.SIGNAL_WEIGHTS.items():
        st.markdown(f"<span class='muted'>{key}: {w:.2f}</span>", unsafe_allow_html=True)
    st.caption("Not investment advice.")

st.title("Market Deployment Gate")
st.caption("Should I be deploying capital right now, and how aggressively?")

h1, h2 = st.columns([1.3, 2])
with h1:
    st.markdown(f"<div class='hero-label'>Deployment score</div>"
                f"<div class='hero-num'>{comp.score:.0f}<small> / 100</small></div>"
                f"{zone_badge(zone['name'])}", unsafe_allow_html=True)
with h2:
    longs = "allowed" if zone["allow_new_longs"] else "<b>blocked</b>"
    scanner = "enabled" if zone["scanner_enabled"] else "<b>disabled</b>"
    as_of = max(r.as_of for r in comp.results.values())
    st.markdown(f"<div class='hero-label'>What this means</div>"
                f"<div class='rule'>Position sizing: <b>{zone['sizing']:.0%}</b></div>"
                f"<div class='rule'>New longs: {longs}</div>"
                f"<div class='rule'>Scanner: {scanner}</div>"
                f"<div class='rule'>{zone['guidance']}</div>"
                f"<div class='muted' style='margin-top:0.6rem'>Exact score {comp.score:.2f} · market data as of "
                f"{as_of:%Y-%m-%d}</div>", unsafe_allow_html=True)

if snap.is_stale():
    st.warning(f"Gate data ends {snap.as_of:%Y-%m-%d}, before the latest completed session. "
               "Click **Refresh data & recalculate** (or run `python run_macro_gate.py`).")
for w in snap.warnings:
    st.warning(w)
if comp.errors:
    st.warning(f"{len(comp.errors)} signal(s) unavailable and excluded; remaining weights re-normalized: "
               + ", ".join(comp.errors))
    for key, err in comp.errors.items():
        st.caption(f"`{key}`: {err}")

st.subheader("Signals")
cols = st.columns(len(comp.results))
for col, (key, r) in zip(cols, comp.results.items()):
    with col:
        st.markdown(f"**{r.name}**", help=r.description)
        show(gauge(r.score))
        st.markdown(f"<div class='muted'>{r.value_label}<br>weight {comp.weights_used[key]:.0%} · "
                    f"as of {r.as_of:%Y-%m-%d}</div>", unsafe_allow_html=True)

st.subheader("Backtest: SPY by deployment zone")
if snap.backtest is None:
    st.warning(snap.backtest_error)
else:
    bt = snap.backtest
    d0, d1 = bt.daily.index[0], bt.daily.index[-1]
    st.caption(f"{d0:%Y-%m-%d} → {d1:%Y-%m-%d} · {len(bt.daily)} trading days · each day's zone comes from "
               "the previous close's score (no look-ahead).")
    show(spy_zone_chart(bt.daily))

    t1, t2 = st.columns([3, 2])
    with t1:
        st.markdown("**SPY returns by zone**")
        table = bt.by_zone.copy()
        table["Days"] = table["Days"].astype(int)
        st.dataframe(
            table, width="stretch",
            column_config={c: st.column_config.NumberColumn(c, format="%.3f" if "daily" in c else "%.1f")
                           for c in table.columns if c != "Days"},
        )
    with t2:
        st.markdown("**Buy & hold vs zone-sized strategy**")
        st.dataframe(bt.summary, width="stretch",
                     column_config={c: st.column_config.NumberColumn(c, format="%.1f") for c in bt.summary.columns})
        st.caption("Strategy = SPY × zone sizing (100% / 60% / 25%); undeployed capital earns 0%.")

hist = snap.history.tail(DISPLAY_DAYS)
show(line_chart(hist["composite"], "Composite deployment score", ".1f", yrange=[0, 100], bands=True))

st.subheader("Signal detail")
tabs = st.tabs([r.name for r in comp.results.values()])
for tab, (key, r) in zip(tabs, comp.results.items()):
    with tab:
        st.caption(r.description)
        title, refs, fmt = RAW_CHART[key]
        h = r.history.dropna(subset=["score"]).tail(DISPLAY_DAYS)
        left, right = st.columns(2)
        with left:
            show(line_chart(h["value"], title, fmt, ref_lines=refs))
        with right:
            show(line_chart(h["score"], f"{r.name} score", ".0f", yrange=[0, 100]))
        if key == crowding.KEY:
            cum = (1 + h[["momentum_ret", "value_ret"]]).cumprod() - 1
            fig = go.Figure()
            for col_name, label, color in (("momentum_ret", "Momentum L/S", SERIES_1),
                                           ("value_ret", "Value proxy L/S", SERIES_2)):
                fig.add_trace(go.Scatter(x=cum.index, y=cum[col_name] * 100, mode="lines", name=label,
                                         line=dict(color=color, width=2),
                                         hovertemplate=f"{label} %{{y:+.1f}}%<extra></extra>"))
            show(layout(fig, "Cumulative factor returns over the window (%)", legend=True))

with st.expander("Table view: daily scores"):
    st.dataframe(snap.history.tail(DISPLAY_DAYS).iloc[::-1].round(1), width="stretch")
