"""Shared dashboard styling and chart helpers (dark theme)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config

BG = "#0b0e17"
SURFACE = "#151a28"
TEXT = "#e6e8ee"
TEXT_MUTED = "#9aa0ad"
GRID = "rgba(255,255,255,0.08)"
SERIES_1 = "#3987e5"   # dark-mode categorical slot 1 (blue)
SERIES_2 = "#d95926"   # dark-mode categorical slot 2 (orange)
# Status colors per zone; always paired with the zone name, never color alone.
ZONE_COLOR = {"FULL DEPLOY": "#0ca30c", "REDUCED": "#fab219", "DEFENSIVE": "#d03b3b"}

CSS = f"""
<style>
  .hero-label {{ color:{TEXT_MUTED}; font-size:0.85rem; letter-spacing:0.08em; text-transform:uppercase; }}
  .hero-num {{ font-size:7.5rem; font-weight:800; line-height:1; color:{TEXT}; font-variant-numeric:tabular-nums; }}
  .hero-num small {{ font-size:2rem; color:{TEXT_MUTED}; font-weight:500; }}
  .zone-badge {{ display:inline-block; padding:0.35rem 0.9rem; border-radius:999px; border:2px solid;
                 font-weight:700; font-size:1.4rem; letter-spacing:0.04em; color:{TEXT}; margin-top:0.6rem; }}
  .rule {{ font-size:1.05rem; margin:0.25rem 0; color:{TEXT}; }}
  .muted {{ color:{TEXT_MUTED}; font-size:0.85rem; }}
  table.blend {{ width:100%; border-collapse:separate; border-spacing:0 6px; font-size:0.92rem; }}
  table.blend th {{ color:{TEXT_MUTED}; font-weight:600; text-align:right; padding:0.3rem 0.6rem;
                    font-size:0.78rem; text-transform:uppercase; letter-spacing:0.05em; }}
  table.blend td {{ background:{SURFACE}; color:{TEXT}; text-align:right; padding:0.55rem 0.6rem;
                    font-variant-numeric:tabular-nums; }}
  table.blend th:nth-child(2), table.blend td:nth-child(2),
  table.blend th:nth-child(3), table.blend td:nth-child(3) {{ text-align:left; }}
  table.blend td:first-child {{ border-radius:8px 0 0 8px; }}
  table.blend td:last-child {{ border-radius:0 8px 8px 0; }}
  table.blend tr.up td {{ box-shadow: 0 -1px 0 #0ca30c inset, 0 1px 0 #0ca30c inset; }}
  table.blend tr.up {{ filter: drop-shadow(0 0 7px rgba(12,163,12,0.75)); }}
  table.blend tr.down td {{ box-shadow: 0 -1px 0 #d03b3b inset, 0 1px 0 #d03b3b inset; }}
  table.blend tr.down {{ filter: drop-shadow(0 0 7px rgba(208,59,59,0.75)); }}
  .delta-up {{ color:#3fd13f; font-weight:700; }}
  .delta-down {{ color:#f06a6a; font-weight:700; }}
  .banner {{ border:1px solid; border-left-width:6px; border-radius:8px; padding:0.8rem 1rem;
             background:{SURFACE}; color:{TEXT}; margin-bottom:1rem; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def zone_badge(name: str) -> str:
    c = ZONE_COLOR[name]
    return f"<span class='zone-badge' style='border-color:{c}'><span style='color:{c}'>●</span> {name}</span>"


def layout(fig: go.Figure, title: str = "", height: int = 300, yrange=None, legend=False) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=TEXT)),
        height=height,
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TEXT_MUTED),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=SURFACE, font_color=TEXT),
        showlegend=legend,
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, font=dict(color=TEXT)),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, range=yrange)
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})


def zone_bands(fig: go.Figure) -> None:
    upper = 100.0
    for zone in config.ZONES:
        fig.add_hrect(y0=zone["min_score"], y1=upper, line_width=0, layer="below",
                      fillcolor=ZONE_COLOR[zone["name"]], opacity=0.10)
        fig.add_annotation(x=0, xref="paper", y=(zone["min_score"] + upper) / 2, yref="y",
                           text=zone["name"], showarrow=False, xanchor="left",
                           font=dict(size=11, color=TEXT_MUTED))
        upper = zone["min_score"]


def line_chart(series: pd.Series, title: str, fmt: str, ref_lines=(), yrange=None, bands=False):
    fig = go.Figure()
    if bands:
        zone_bands(fig)
    for value, label in ref_lines:
        fig.add_hline(y=value, line_dash="dot", line_width=1, line_color="rgba(200,200,210,0.45)",
                      annotation_text=label, annotation_position="top right",
                      annotation_font=dict(size=10, color=TEXT_MUTED))
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name=title,
                             line=dict(color=SERIES_1, width=2),
                             hovertemplate=f"%{{y:{fmt}}}<extra></extra>"))
    return layout(fig, title, yrange=yrange)
