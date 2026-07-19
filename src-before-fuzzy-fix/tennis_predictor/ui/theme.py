from __future__ import annotations

import streamlit as st

from tennis_predictor.storage import (
    get_wallet_summary,
    initialize_paper_trading_db,
    set_wallet_balance,
)

from .data import clear_dashboard_cache, load_app_config
from .paths import DashboardPaths


GLOBAL_CSS = """
<style>
:root {
  --black: #0d0d0d;
  --panel: #1a1a1a;
  --panel-soft: #141414;
  --gold: #D4AF37;
  --gold-soft: rgba(212, 175, 55, 0.13);
  --white: #ffffff;
  --muted: #b8b8b8;
  --border: rgba(212, 175, 55, 0.55);
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--black) !important;
  color: var(--white) !important;
}
[data-testid="stHeader"] { background: rgba(13,13,13,.92) !important; }
[data-testid="stSidebar"] {
  background: #111111 !important;
  border-right: 1px solid rgba(212,175,55,.35);
}
.block-container { max-width: 1500px; padding-top: 1.6rem; padding-bottom: 4rem; }

h1, h2, h3, h4, h5, h6,
[data-testid="stMarkdownContainer"] strong { color: var(--white); }
.eyebrow, .section-label { color: var(--gold) !important; }
p, label, [data-testid="stCaptionContainer"] { color: var(--muted); }

.hero-shell {
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 2rem 2.2rem;
  margin-bottom: 1.25rem;
  background: linear-gradient(135deg, rgba(212,175,55,.10), rgba(26,26,26,.96) 48%), var(--panel);
  box-shadow: 0 18px 55px rgba(0,0,0,.35);
}
.hero-shell h1 { margin: .25rem 0 .5rem; font-size: clamp(2.1rem, 4vw, 3.8rem); }
.hero-shell p { color: #d0d0d0; font-size: 1.02rem; max-width: 950px; }
.eyebrow { font-weight: 850; letter-spacing: .16em; font-size: .74rem; }

.metric-card, [data-testid="stMetric"] {
  background: var(--panel) !important;
  border: 1px solid var(--border) !important;
  border-radius: 15px !important;
  padding: 1rem 1.1rem !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.025), 0 10px 30px rgba(0,0,0,.22);
}
[data-testid="stMetricLabel"] { color: var(--gold) !important; }
[data-testid="stMetricValue"] { color: var(--white) !important; font-weight: 800; }
.metric-label { color: var(--gold); font-size: .76rem; text-transform: uppercase; letter-spacing: .08em; }
.metric-value { color: var(--white); font-size: 1.9rem; font-weight: 800; margin-top: .3rem; }
.metric-detail { color: var(--muted); font-size: .82rem; margin-top: .28rem; }

div[data-testid="stVerticalBlockBorderWrapper"] > div {
  background: var(--panel-soft);
  border-color: rgba(212,175,55,.35) !important;
  border-radius: 16px;
}
.callout {
  padding: 1rem 1.1rem;
  border-radius: 12px;
  border: 1px solid rgba(212,175,55,.35);
  background: var(--panel);
  color: #dedede;
}
.small-muted { color: var(--muted); font-size: .84rem; }

.stButton > button,
.stFormSubmitButton > button,
button[kind="primary"],
button[kind="secondary"] {
  background: #0d0d0d !important;
  color: var(--gold) !important;
  border: 2px solid var(--gold) !important;
  border-radius: 10px !important;
  font-weight: 800 !important;
  box-shadow: none !important;
}
.stButton > button:hover,
.stFormSubmitButton > button:hover,
button[kind="primary"]:hover,
button[kind="secondary"]:hover {
  background: var(--gold) !important;
  color: #0d0d0d !important;
  border-color: var(--gold) !important;
}

[data-baseweb="input"] > div,
[data-baseweb="select"] > div,
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
  background: #121212 !important;
  color: var(--white) !important;
  border-color: rgba(212,175,55,.38) !important;
}

.status-ready, .status-pass, .status-candidate, .status-warning {
  display: inline-block;
  padding: .38rem .68rem;
  border-radius: 999px;
  font-size: .74rem;
  font-weight: 850;
  letter-spacing: .04em;
  border: 1px solid;
}
.status-ready { color: var(--gold); border-color: var(--gold); background: var(--gold-soft); }
.status-pass { color: #d4d4d4; border-color: #666; background: rgba(255,255,255,.04); }
.status-candidate { color: #0d0d0d; border-color: var(--gold); background: var(--gold); }
.status-warning { color: #ffd98a; border-color: #9c7621; background: rgba(212,175,55,.10); }

.price-box {
  padding: .9rem 1rem;
  border: 1px solid rgba(212,175,55,.35);
  background: #111;
  border-radius: 12px;
}
.price-title { color: var(--gold); font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; }
.price-value { color: var(--white); font-size: 1.25rem; font-weight: 800; }

[data-testid="stDataFrame"] {
  border: 1px solid rgba(212,175,55,.35);
  border-radius: 12px;
  overflow: hidden;
}
hr { border-color: rgba(212,175,55,.20) !important; }
a { color: var(--gold) !important; }

.matchup-shell {
  position: relative;
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  align-items: center;
  gap: 1rem;
  padding: 1.35rem 1.5rem 2rem;
  margin: 1rem 0 1.25rem;
  border: 1px solid rgba(212,175,55,.42);
  border-radius: 16px;
  background: linear-gradient(135deg, rgba(212,175,55,.08), rgba(18,18,18,.96));
  overflow: hidden;
}
.player-side { min-width: 0; }
.player-two { text-align: right; }
.player-name {
  color: var(--white);
  font-size: clamp(1.05rem, 2vw, 1.55rem);
  font-weight: 820;
  line-height: 1.2;
}
.player-prob {
  color: var(--gold);
  font-size: clamp(1.8rem, 4vw, 3rem);
  font-weight: 900;
  margin-top: .25rem;
}
.versus {
  color: #777;
  font-size: .8rem;
  font-weight: 900;
  letter-spacing: .14em;
}
.probability-track {
  position: absolute;
  left: 1.5rem;
  right: 1.5rem;
  bottom: .85rem;
  height: 7px;
  border-radius: 999px;
  background: #303030;
  overflow: hidden;
}
.probability-fill {
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, #A88622, var(--gold));
}
.prediction-headline {
  padding: 1.2rem 1.35rem;
  margin-top: 1rem;
  border-left: 4px solid var(--gold);
  border-radius: 0 14px 14px 0;
  background: #171717;
}
.prediction-kicker, .action-label {
  color: var(--gold);
  font-size: .72rem;
  font-weight: 900;
  letter-spacing: .15em;
}
.prediction-favorite {
  color: var(--white);
  font-size: clamp(1.8rem, 4vw, 2.8rem);
  font-weight: 900;
  margin-top: .2rem;
}
.prediction-confidence { color: #e8e8e8; font-size: 1.02rem; }
.prediction-warning { color: #999; font-size: .82rem; margin-top: .35rem; }
.decision-explainer {
  padding: 1rem 1.1rem;
  margin-bottom: 1rem;
  border: 1px solid rgba(212,175,55,.28);
  border-radius: 12px;
  background: #141414;
  color: #d6d6d6;
}
.action-banner {
  margin: 1.1rem 0 1.4rem;
  padding: 1.15rem 1.25rem;
  border-radius: 14px;
  border: 1px solid;
}
.action-banner.candidate {
  border-color: var(--gold);
  background: linear-gradient(135deg, rgba(212,175,55,.18), #171717 55%);
}
.action-banner.pass {
  border-color: #555;
  background: #171717;
}
.action-title {
  color: var(--white);
  font-size: 1.35rem;
  font-weight: 900;
  margin: .2rem 0;
}
.action-copy { color: #c8c8c8; }

[data-baseweb="radio"] input:checked + div,
[data-baseweb="checkbox"] input:checked + div {
  border-color: var(--gold) !important;
  background-color: var(--gold) !important;
}

@media (max-width: 760px) {
  .matchup-shell { grid-template-columns: 1fr; text-align: left; }
  .player-two { text-align: left; }
  .versus { display: none; }
}

</style>
"""


def apply_global_styles() -> None:
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def render_sidebar() -> None:
    st.sidebar.markdown("## 🎾 Tennis Edge Lab")
    st.sidebar.caption("Polymarket execution · leakage-safe research")

    config = load_app_config()
    db_path = config.paper_trading.database_path
    initialize_paper_trading_db(db_path)
    summary = get_wallet_summary(db_path)
    with st.sidebar.form("bankroll-editor", border=False):
        bankroll = st.number_input(
            "Current Bankroll ($)",
            min_value=0.0,
            value=float(summary.balance),
            step=10.0,
            format="%.2f",
        )
        apply_balance = st.form_submit_button("Update paper bankroll", width="stretch")
    if apply_balance:
        set_wallet_balance(db_path, float(bankroll), note="Sidebar bankroll update")
        st.success(f"Paper bankroll updated to ${float(bankroll):,.2f}.")
        st.rerun()

    defaults = DashboardPaths.defaults()
    with st.sidebar.expander("Artifact paths", expanded=False):
        model = st.text_input("Model", value=str(defaults.model), key="dashboard_model_path")
        metrics = st.text_input("Metrics", value=str(defaults.metrics), key="dashboard_metrics_path")
        predictions = st.text_input(
            "Predictions", value=str(defaults.predictions), key="dashboard_predictions_path"
        )
        st.session_state["dashboard_paths"] = DashboardPaths.defaults().with_overrides(
            model=model, metrics=metrics, predictions=predictions
        )

    paths = st.session_state.get("dashboard_paths", defaults)
    for label, exists in paths.status().items():
        icon = "●" if exists else "○"
        st.sidebar.markdown(f"{icon} **{label}**", help=str(getattr(paths, label.lower())))

    if st.sidebar.button("Reload artifacts", width="stretch"):
        clear_dashboard_cache()
        st.rerun()

    st.sidebar.divider()
    st.sidebar.caption("Local paper trading only. No real orders are submitted.")
