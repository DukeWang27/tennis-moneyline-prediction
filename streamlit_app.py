from __future__ import annotations

import streamlit as st

from tennis_predictor.services import settle_recent_pending_bets
from tennis_predictor.ui.data import load_app_config
from tennis_predictor.ui.pages import (
    render_backtest_lab,
    render_home,
    render_match_explorer,
    render_model_guide,
    render_paper_bets,
    render_paper_metric_banner,
    render_predictor,
    render_today_matches,
)
from tennis_predictor.ui.theme import apply_global_styles, render_sidebar
from tennis_predictor.ui.identity_admin import render_identity_review

st.set_page_config(
    page_title="Tennis Edge Lab",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=300, show_spinner=False)
def _startup_settlement() -> dict[str, int]:
    config = load_app_config()
    return settle_recent_pending_bets(config, days_back=2)


apply_global_styles()
try:
    settled = _startup_settlement()
    if any(settled.values()):
        st.toast(
            f"Paper bets settled: {settled['won']} won, "
            f"{settled['lost']} lost, {settled['void']} void."
        )
except Exception as exc:
    # Live settlement must never prevent the research dashboard from opening.
    st.session_state["startup_settlement_error"] = str(exc)

render_sidebar()
render_paper_metric_banner()

pages = {
    "Dashboard": [
        st.Page(render_home, title="Overview", icon=":material/dashboard:", default=True),
        st.Page(render_predictor, title="Match Predictor", icon=":material/analytics:"),
        st.Page(render_today_matches, title="Polymarket Scanner", icon=":material/event:"),
    ],
    "Paper Trading": [
        st.Page(render_paper_bets, title="Paper-Bet Tracker", icon=":material/account_balance_wallet:"),
    ],
    "Research": [
        st.Page(render_identity_review, title="Alias Review Queue", icon=":material/person_search:"),
        st.Page(render_backtest_lab, title="Backtest Lab", icon=":material/query_stats:"),
        st.Page(render_match_explorer, title="Match Explorer", icon=":material/table_view:"),
        st.Page(render_model_guide, title="Model Guide", icon=":material/menu_book:"),
    ],
}

navigation = st.navigation(pages, position="sidebar")
navigation.run()
