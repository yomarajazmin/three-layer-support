"""Streamlit dashboard entry point.

Run with:  streamlit run app.py
  Page 1: Macro deployment gate ("Should I be deploying capital right now, and how aggressively?")
  Page 2: Gated multi-factor stock scanner
  Page 3: Claude analyst layer (blended quant + fundamental ranking)
"""

import streamlit as st

from views.theme import inject_css

st.set_page_config(page_title="Market Deployment Gate", page_icon="📊", layout="wide")
inject_css()

nav = st.navigation([
    st.Page("views/macro_gate.py", title="Macro Gate", icon="🧭", default=True),
    st.Page("views/scanner_page.py", title="Stock Scanner", icon="🔎", url_path="scanner"),
    st.Page("views/analyst_page.py", title="AI Analyst", icon="🧠", url_path="analyst"),
])
nav.run()
