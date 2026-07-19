from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .components import (
    format_number,
    format_percent,
    hero,
    probability_gauge,
    probability_matchup,
    render_metric_row,
    status_badge,
)
from .data import (
    load_app_config,
    load_metrics,
    load_predictions,
    load_system,
    player_options,
    safe_load,
)
from .view_models import clean_unresolved_fixtures
from .paths import DashboardPaths
from tennis_predictor.data.names import normalize_name
from tennis_predictor.market import PolymarketClient, betting_signal, fractional_kelly
from tennis_predictor.services import build_live_board
from tennis_predictor.storage import (
    available_history_dates,
    get_wallet_summary,
    history_matches,
    initialize_live_store,
    initialize_paper_trading_db,
    list_bets,
    place_paper_bet,
    save_player_alias,
    save_tournament_metadata,
    seed_walkforward_ledger,
    settle_pending_bets_from_results,
    upsert_match_history,
)


def _paths() -> DashboardPaths:
    return st.session_state.get("dashboard_paths", DashboardPaths.defaults())


def _empty_state(title: str, message: str, command: str | None = None) -> None:
    st.warning(title)
    st.markdown(f'<div class="callout">{message}</div>', unsafe_allow_html=True)
    if command:
        st.code(command, language="bash")


def _overall_metric(metrics: pd.DataFrame, model: str, filter_view: str = "all") -> pd.Series | None:
    match = metrics[
        (metrics["filter_view"] == filter_view)
        & (metrics["segment_type"] == "full")
        & (metrics["segment"] == model)
    ]
    return None if match.empty else match.iloc[0]


def render_home() -> None:
    hero(
        "Turn match history into a decision dashboard.",
        "Inspect calibrated model performance, compare it with Elo, and move from research output to a usable prediction product.",
    )
    paths = _paths()
    metrics, metrics_error = safe_load(paths.metrics, load_metrics)
    predictions, predictions_error = safe_load(paths.predictions, load_predictions)

    if metrics_error:
        _empty_state(
            "Backtest metrics are not available yet.",
            metrics_error,
            "tennis-backtest --config config.yaml --output-dir outputs/backtest",
        )
        return

    boosted = _overall_metric(metrics, "boosted")
    elo = _overall_metric(metrics, "elo_baseline")
    filtered = _overall_metric(metrics, "boosted", "filtered")

    if boosted is not None:
        render_metric_row(
            [
                ("Walk-forward matches", f"{int(boosted['matches']):,}", "Strictly historical test rows"),
                ("Boosted accuracy", format_percent(boosted["accuracy"]), "Calibrated LightGBM"),
                ("Brier score", format_number(boosted["brier"], 4), "Lower is better"),
                ("Log loss", format_number(boosted["log_loss"], 4), "Confidence-sensitive"),
            ]
        )

    st.markdown("### Model comparison")
    full = metrics[(metrics["filter_view"] == "all") & (metrics["segment_type"] == "full")].copy()
    if not full.empty:
        full["Model"] = full["segment"].replace({"boosted": "Boosted", "elo_baseline": "Elo baseline"})
        comparison = full.melt(
            id_vars=["Model"],
            value_vars=["accuracy", "brier", "log_loss"],
            var_name="Metric",
            value_name="Value",
        )
        figure = px.bar(
            comparison,
            x="Metric",
            y="Value",
            color="Model",
            barmode="group",
            text_auto=".3f",
            color_discrete_sequence=["#b7ff3c", "#46d7ff"],
        )
        figure.update_layout(height=390, legend_title_text="", margin=dict(l=10, r=10, t=25, b=10))
        st.plotly_chart(figure, width="stretch")

    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("### Accuracy by tournament tier")
        tiers = metrics[
            (metrics["filter_view"] == "all") & (metrics["segment_type"] == "tier")
        ].copy()
        if not tiers.empty:
            split = tiers["segment"].str.split("|", regex=False, n=1, expand=True)
            tiers["Model"] = split[0].replace({"boosted": "Boosted", "elo_baseline": "Elo"})
            tiers["Tier"] = split[1]
            figure = px.bar(
                tiers,
                x="Tier",
                y="accuracy",
                color="Model",
                barmode="group",
                text_auto=".1%",
                color_discrete_sequence=["#b7ff3c", "#46d7ff"],
            )
            figure.update_yaxes(tickformat=".0%", range=[0.5, max(0.75, float(tiers["accuracy"].max()) + 0.03)])
            figure.update_layout(height=410, legend_title_text="", margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(figure, width="stretch")
    with right:
        st.markdown("### What changed with filters?")
        if boosted is not None and filtered is not None:
            delta = filtered["accuracy"] - boosted["accuracy"]
            retained = filtered["matches"] / boosted["matches"]
            render_metric_row(
                [
                    ("Accuracy change", f"{delta:+.2%}", "Filtered minus full sample"),
                    ("Matches retained", format_percent(retained), "Rows still eligible"),
                ]
            )
            st.markdown(
                "The filter layer should be treated as an experiment, not a truth rule. "
                "A small accuracy increase can still be noise, and calibration may move differently."
            )
        if elo is not None and boosted is not None:
            st.markdown("### Current read")
            accuracy_winner = "Boosted" if boosted["accuracy"] > elo["accuracy"] else "Elo"
            logloss_winner = "Boosted" if boosted["log_loss"] < elo["log_loss"] else "Elo"
            st.markdown(
                f"- **Accuracy leader:** {accuracy_winner}\n"
                f"- **Log-loss leader:** {logloss_winner}\n"
                f"- **Market bets available:** {'Yes' if int(boosted.get('bets', 0)) else 'No — odds are not loaded'}"
            )

    if predictions_error is None and predictions is not None and not predictions.empty:
        st.markdown("### Testing coverage")
        coverage = predictions.groupby([predictions["date"].dt.to_period("Q").astype(str), "model"]).size().reset_index(name="matches")
        coverage.columns = ["Quarter", "Model", "Matches"]
        figure = px.area(
            coverage,
            x="Quarter",
            y="Matches",
            color="Model",
            color_discrete_sequence=["#46d7ff", "#b7ff3c"],
        )
        figure.update_layout(height=300, legend_title_text="", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(figure, width="stretch")


def _context_value(value: str) -> bool | None:
    return {"Unknown": None, "No": False, "Yes": True}[value]


def _default_player_index(options: list[tuple[str, str]], preferred: str, fallback: int = 0) -> int:
    names = [name for name, _ in options]
    try:
        return names.index(preferred)
    except ValueError:
        return min(fallback, max(len(names) - 1, 0))



def _render_historical_replay(paths: DashboardPaths) -> None:
    """Replay only probabilities captured before the selected result was applied."""
    config = load_app_config()
    db_path = config.paper_trading.database_path
    initialize_live_store(db_path)

    # Seed the dynamic history store from the project's existing artifacts once.
    dates = available_history_dates(db_path)
    if not dates:
        processed_path = Path(config.data.processed_path)
        if processed_path.exists():
            upsert_match_history(db_path, pd.read_parquet(processed_path))
        predictions, error = safe_load(paths.predictions, load_predictions)
        if error is None and predictions is not None and not predictions.empty:
            seed_walkforward_ledger(db_path, predictions)
        dates = available_history_dates(db_path)

    if not dates:
        st.warning("No historical matches are stored yet. Run tennis-sync first.")
        return

    st.success(
        "Leakage-safe replay: probabilities come only from the saved walk-forward "
        "ledger or from a sequential live sync that scored each pre-match snapshot "
        "before updating Elo, form, rest, and H2H."
    )

    controls = st.columns([0.8, 1.1, 1.5])
    with controls[0]:
        tour = st.radio(
            "Tour", ["ATP", "WTA"], horizontal=True, key="historical-db-tour"
        )
    with controls[1]:
        chosen_date = st.selectbox(
            "Historical match date",
            options=[value.date() for value in reversed(dates)],
            key="historical-db-date",
        )
    with controls[2]:
        player_search = st.text_input(
            "Optional player search",
            placeholder="Type a surname",
            key="historical-db-player-search",
        )

    day_matches = history_matches(db_path, match_date=chosen_date)
    if day_matches.empty:
        st.info("No stored matches are available on that date.")
        return
    filtered = day_matches[
        day_matches["tour"].fillna("").astype(str).str.upper().eq(tour)
    ].copy()
    if player_search.strip():
        query = player_search.strip().casefold()
        filtered = filtered[
            filtered["player1"].astype(str).str.casefold().str.contains(query, regex=False)
            | filtered["player2"].astype(str).str.casefold().str.contains(query, regex=False)
        ]
    if filtered.empty:
        st.info(f"No stored {tour} matches match those filters on {chosen_date}.")
        return

    tournaments = ["All tournaments"] + sorted(
        filtered["tournament"].fillna("Unknown").astype(str).unique().tolist()
    )
    tournament = st.selectbox(
        "Tournament", tournaments, key="historical-db-tournament"
    )
    if tournament != "All tournaments":
        filtered = filtered[
            filtered["tournament"].fillna("Unknown").astype(str).eq(tournament)
        ]
    if filtered.empty:
        st.info("No matches remain after the tournament filter.")
        return

    filtered["label"] = (
        filtered["player1"].astype(str)
        + " vs "
        + filtered["player2"].astype(str)
        + " · "
        + filtered["tournament"].fillna("Unknown").astype(str)
        + " · "
        + filtered["surface"].fillna("Unknown").astype(str)
    )
    selected_id = st.selectbox(
        "Historical match",
        options=filtered["match_id"].astype(str).tolist(),
        format_func=dict(zip(filtered["match_id"].astype(str), filtered["label"])).get,
        key="historical-db-match",
    )
    match = filtered[filtered["match_id"].astype(str).eq(str(selected_id))].iloc[0]

    p_model = pd.to_numeric(pd.Series([match.get("p_model1")]), errors="coerce").iloc[0]
    p_elo = pd.to_numeric(pd.Series([match.get("p_elo1")]), errors="coerce").iloc[0]
    prediction_status = str(match.get("prediction_status") or "NOT STORED")
    if pd.isna(p_model):
        st.warning(
            "This match is stored, but no leakage-safe boosted probability is available. "
            + (
                "Its surface was unknown, so the result updated general player state but "
                "was not passed through the surface-dependent model."
                if str(match.get("surface", "Unknown")) == "Unknown"
                else "It was imported before the sequential prediction ledger was enabled. "
                     "Use tennis-rebuild-live with the frozen pre-live model to recreate it safely."
            )
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "Field": ["Date", "Tournament", "Round", "Tour", "Surface", "Tier", "Prediction status"],
                    "Value": [
                        pd.Timestamp(match["date"]).date().isoformat(),
                        match.get("tournament", "Unknown"),
                        match.get("round", "Unknown"),
                        match.get("tour", "Unknown"),
                        match.get("surface", "Unknown"),
                        match.get("tier", "Unknown"),
                        prediction_status,
                    ],
                }
            ),
            hide_index=True,
            width="stretch",
        )
    else:
        probability = float(p_model)
        probability_matchup(str(match["player1"]), str(match["player2"]), probability)
        chart_column, metric_column = st.columns([1.1, 1])
        with chart_column:
            st.plotly_chart(
                probability_gauge(str(match["player1"]), probability), width="stretch"
            )
        with metric_column:
            render_metric_row(
                [
                    ("Boosted model", format_percent(probability), "Saved before result update"),
                    (
                        "Elo baseline",
                        format_percent(p_elo) if pd.notna(p_elo) else "—",
                        "Same pre-match state",
                    ),
                    ("Confidence", format_percent(max(probability, 1 - probability)), prediction_status),
                ]
            )
            if pd.notna(p_elo):
                disagreement = abs(probability - float(p_elo))
                st.progress(
                    min(disagreement / 0.25, 1.0),
                    text=f"Model disagreement: {disagreement:.1%}",
                )

    reveal = st.checkbox(
        "Reveal the actual result",
        value=False,
        key=f"historical-db-reveal:{selected_id}",
    )
    if not reveal:
        st.info("Result hidden to reduce hindsight bias.")
        return
    actual = pd.to_numeric(
        pd.Series([match.get("ledger_actual", match.get("player1_won"))]), errors="coerce"
    ).iloc[0]
    if pd.isna(actual):
        st.warning("A final winner has not been stored for this match.")
        return
    actual_winner = str(match["player1"] if int(actual) == 1 else match["player2"])
    result_columns = st.columns(3)
    result_columns[0].metric("Actual winner", actual_winner)
    if pd.notna(p_model):
        predicted_winner = str(match["player1"] if float(p_model) >= 0.5 else match["player2"])
        result_columns[1].metric("Model pick", predicted_winner)
        result_columns[2].metric(
            "Outcome", "CORRECT" if predicted_winner == actual_winner else "INCORRECT"
        )
    else:
        result_columns[1].metric("Model pick", "Not stored")
        result_columns[2].metric("Outcome", "Not scored")


def _predictor_price_evaluation(
    *,
    player: str,
    probability: float,
    ask_cents: float,
    bankroll: float,
    config,
) -> dict[str, object] | None:
    if not 0 < float(ask_cents) < 100:
        return None
    ask = float(ask_cents) / 100.0
    decimal_odds = 1.0 / ask
    recommendation = fractional_kelly(
        probability,
        decimal_odds,
        bankroll,
        multiplier=config.paper_trading.kelly_multiplier,
        max_fraction=config.paper_trading.max_kelly_fraction,
    )
    return {
        "player": player,
        "ask": ask,
        "decimal_odds": decimal_odds,
        "recommendation": recommendation,
        "signal": betting_signal(
            recommendation,
            min_edge=config.paper_trading.min_edge,
        ),
    }


def _render_predictor_betting_panel(
    *,
    result: dict[str, object],
    prediction_context: dict[str, object],
    allow_betting: bool = True,
) -> None:
    config = load_app_config()
    wallet = get_wallet_summary(config.paper_trading.database_path)
    player1 = str(result["player1"])
    player2 = str(result["player2"])
    p1 = float(result["player1_win_probability"])
    p2 = 1.0 - p1

    st.markdown("## Betting decision")
    if not allow_betting:
        st.markdown(
            '<div class="action-banner pass"><div class="action-label">NO BET — LIMITED DATA</div>'
            '<div class="action-title">This matchup does not meet the sample-size gate.</div>'
            '<div class="action-copy">The probability may be viewed for research, but edge, Kelly, '
            'and paper-bet controls are disabled. Pick a matchup marked SUFFICIENT SAMPLE.</div></div>',
            unsafe_allow_html=True,
        )
        return
    st.markdown(
        '<div class="decision-explainer">A win probability alone cannot tell you to bet. '
        'Enter the current <strong>best Polymarket ask</strong> for each player. The app '
        'then compares your model probability with the price you would actually pay.</div>',
        unsafe_allow_html=True,
    )

    price_columns = st.columns(2)
    ask1_cents = price_columns[0].number_input(
        f"{player1} best ask (cents)",
        min_value=0.0,
        max_value=99.9,
        value=0.0,
        step=0.5,
        help="Enter 56 for a $0.56 Polymarket share. Leave 0 when no price is available.",
        key="predictor-ask-player1",
    )
    ask2_cents = price_columns[1].number_input(
        f"{player2} best ask (cents)",
        min_value=0.0,
        max_value=99.9,
        value=0.0,
        step=0.5,
        help="Enter 44 for a $0.44 Polymarket share. Leave 0 when no price is available.",
        key="predictor-ask-player2",
    )

    evaluations = [
        _predictor_price_evaluation(
            player=player1,
            probability=p1,
            ask_cents=ask1_cents,
            bankroll=wallet.balance,
            config=config,
        ),
        _predictor_price_evaluation(
            player=player2,
            probability=p2,
            ask_cents=ask2_cents,
            bankroll=wallet.balance,
            config=config,
        ),
    ]

    side_columns = st.columns(2)
    for column, player, probability, evaluation in zip(
        side_columns,
        (player1, player2),
        (p1, p2),
        evaluations,
        strict=True,
    ):
        with column:
            with st.container(border=True):
                st.markdown(f"### {player}")
                st.metric("Model win probability", f"{probability:.1%}")
                if evaluation is None:
                    st.markdown(_status_badge("PASS"), unsafe_allow_html=True)
                    st.caption("No executable ask entered, so there is no price-based decision.")
                    continue
                recommendation = evaluation["recommendation"]
                signal = str(evaluation["signal"])
                st.metric(
                    "Executable price",
                    f"{float(evaluation['ask']) * 100:.1f}¢",
                    delta=f"Decimal {float(evaluation['decimal_odds']):.3f}",
                    delta_color="off",
                )
                metric_columns = st.columns(2)
                metric_columns[0].metric("Model edge", f"{recommendation.edge:+.1%}")
                metric_columns[1].metric(
                    "Quarter-Kelly",
                    _money(recommendation.stake),
                    delta=f"{recommendation.applied_fraction:.1%} bankroll",
                    delta_color="off",
                )
                st.markdown(_status_badge(signal), unsafe_allow_html=True)

    valid_evaluations = [item for item in evaluations if item is not None]
    candidates = [
        item
        for item in valid_evaluations
        if item["signal"] == "PAPER-BET CANDIDATE"
    ]
    if candidates:
        best = max(
            candidates,
            key=lambda item: (
                item["recommendation"].edge,
                item["recommendation"].stake,
            ),
        )
        recommendation = best["recommendation"]
        st.markdown(
            f'<div class="action-banner candidate"><div class="action-label">PAPER-BET CANDIDATE</div>'
            f'<div class="action-title">{best["player"]} @ {float(best["decimal_odds"]):.3f}</div>'
            f'<div class="action-copy">Model edge {recommendation.edge:+.1%}. Suggested paper stake '
            f'{_money(recommendation.stake)} ({recommendation.applied_fraction:.1%} of the current '
            f'{_money(wallet.balance)} bankroll).</div></div>',
            unsafe_allow_html=True,
        )
    elif valid_evaluations:
        st.markdown(
            '<div class="action-banner pass"><div class="action-label">PASS</div>'
            '<div class="action-title">Neither side clears the edge and Kelly rules.</div>'
            '<div class="action-copy">The model may favor one player to win while the market '
            'price is still too expensive to justify a paper-bet signal.</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("Enter at least one current Polymarket ask to receive a PASS or PAPER-BET CANDIDATE decision.")
        return

    st.markdown("### Paper-bet ticket")
    available_players = [str(item["player"]) for item in valid_evaluations]
    selected_player = st.selectbox(
        "Bet on",
        available_players,
        key="predictor-paper-side",
    )
    selected = next(
        item for item in valid_evaluations if item["player"] == selected_player
    )
    selected_recommendation = selected["recommendation"]
    suggested_stake = (
        selected_recommendation.stake
        if selected_recommendation.stake > 0
        else min(1.0, wallet.balance)
    )
    ticket_columns = st.columns([1, 1, 1.25])
    stake = ticket_columns[0].number_input(
        "Paper stake ($)",
        min_value=0.01,
        max_value=max(float(wallet.balance), 0.01),
        value=max(min(float(suggested_stake), float(wallet.balance)), 0.01),
        step=1.0,
        format="%.2f",
        key="predictor-paper-stake",
    )
    ticket_columns[1].metric("Available bankroll", _money(wallet.balance))
    signal = str(selected["signal"])
    button_label = (
        "Simulate recommended bet"
        if signal == "PAPER-BET CANDIDATE"
        else "Log paper bet anyway"
    )
    with ticket_columns[2]:
        st.caption(
            "This records a simulated position only."
            if signal == "PAPER-BET CANDIDATE"
            else "The model says PASS; logging it is an intentional research override."
        )
        submit_bet = st.button(
            button_label,
            width="stretch",
            key="predictor-submit-paper-bet",
        )
    if not submit_bet:
        return

    match_date = pd.Timestamp(prediction_context["match_date"]).date().isoformat()
    tour = str(prediction_context["tour"])
    side_slug = normalize_name(selected_player).replace(" ", "-")
    matchup_slug = "-vs-".join(
        sorted(
            [
                normalize_name(player1).replace(" ", "-"),
                normalize_name(player2).replace(" ", "-"),
            ]
        )
    )
    fixture_id = f"manual:{match_date}:{tour.casefold()}:{matchup_slug}:{side_slug}"
    opponent = player2 if selected_player == player1 else player1
    selected_probability = p1 if selected_player == player1 else p2
    try:
        place_paper_bet(
            config.paper_trading.database_path,
            fixture_id=fixture_id,
            matchup=f"{player1} vs {player2}",
            bet_on=selected_player,
            odds=float(selected["decimal_odds"]),
            sportsbook="Polymarket (manual best ask)",
            stake=float(stake),
            p_model=selected_probability,
            provider_market_id=None,
            token_id=None,
            opponent=opponent,
            tournament=str(prediction_context["tier"]),
            commence_time=prediction_context["match_date"],
            edge=selected_recommendation.edge,
            kelly_fraction=selected_recommendation.applied_fraction,
        )
    except Exception as exc:
        st.error(str(exc))
    else:
        st.success(
            f"Paper bet logged: {_money(stake)} on {selected_player} "
            f"@ {float(selected['decimal_odds']):.3f}."
        )
        st.rerun()


def _render_sample_quality_panel(sample: dict[str, object], *, surface: str) -> str:
    status = str(sample.get("status") or "UNKNOWN")
    label = {
        "SUFFICIENT": "SUFFICIENT SAMPLE",
        "LIMITED": "LIMITED SAMPLE / NO BET",
        "INSUFFICIENT": "INSUFFICIENT SAMPLE / NO PREDICTION",
    }.get(status, "SAMPLE QUALITY UNKNOWN")
    st.markdown(_status_badge(label), unsafe_allow_html=True)
    st.caption(str(sample.get("reason") or "No sample-quality explanation was returned."))
    rows = []
    for key in ("player1", "player2"):
        player = sample.get(key) or {}
        if not isinstance(player, dict):
            continue
        rows.append(
            {
                "Player": player.get("name", "Unknown"),
                "Recorded matches": int(player.get("total_matches") or 0),
                f"{surface} matches": int(player.get("surface_matches") or 0),
                "Last recorded match": player.get("last_match_date") or "Unknown",
                "Days stale": player.get("staleness_days") if player.get("staleness_days") is not None else "—",
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    return status


def _render_cached_alternatives() -> None:
    frames = []
    for key, value in st.session_state.items():
        if not str(key).startswith("polymarket-first-board:"):
            continue
        markets = getattr(value, "markets", None)
        if isinstance(markets, pd.DataFrame) and not markets.empty:
            frames.append(markets.copy())
    if not frames:
        st.info("Open Polymarket Scanner and refresh the selected day to find matchups with sufficient data.")
        return
    markets = pd.concat(frames, ignore_index=True, sort=False).drop_duplicates("market_id")
    eligible = markets[
        markets.get("sample_status", pd.Series("", index=markets.index)).eq("SUFFICIENT")
        & markets.get("model_status", pd.Series("", index=markets.index)).eq("READY")
    ].copy()
    if eligible.empty:
        st.info("The cached scanner has no matchup that currently meets the sufficient-sample gate.")
        return
    eligible["best_edge"] = eligible[["edge1", "edge2"]].apply(pd.to_numeric, errors="coerce").max(axis=1)
    eligible = eligible.sort_values("best_edge", ascending=False).head(5)
    st.markdown("### Better-covered alternatives")
    display = eligible[["display_title", "tournament", "surface", "total_matches1", "total_matches2", "best_edge"]].copy()
    display.columns = ["Match", "Tournament", "Surface", "P1 matches", "P2 matches", "Best model edge"]
    st.dataframe(display, hide_index=True, width="stretch", column_config={"Best model edge": st.column_config.NumberColumn(format="%+.1f%%")})


def render_predictor() -> None:
    hero(
        "Match Predictor",
        "Generate a matchup probability, compare it with an executable market price, and log the decision to paper trading.",
        eyebrow="PREDICTION WORKBENCH",
    )
    paths = _paths()
    mode = st.radio(
        "Prediction mode",
        options=["Current / upcoming", "Historical replay — leakage safe"],
        horizontal=True,
        key="prediction-mode",
    )
    if mode == "Historical replay — leakage safe":
        _render_historical_replay(paths)
        return
    system, error = safe_load(paths.model, load_system)
    if error:
        _empty_state(
            "A trained model is required.",
            error,
            "tennis-backtest --config config.yaml --output-dir outputs/backtest",
        )
        return

    top_left, top_right = st.columns([2.1, 1])
    with top_left:
        tour = st.radio("Tour", options=["ATP", "WTA"], horizontal=True)
    with top_right:
        as_of = getattr(system.feature_engineer.state, "as_of", None)
        st.metric(
            "Model state",
            as_of.date().isoformat() if as_of is not None else "Unknown",
        )

    options = player_options(system, tour)
    if len(options) < 2:
        st.error(f"The trained system does not contain enough {tour} players.")
        return
    names = [name for name, _ in options]

    with st.form("prediction_form"):
        player_columns = st.columns(2)
        with player_columns[0]:
            p1 = st.selectbox(
                "Player 1",
                names,
                index=_default_player_index(options, "Carlos Alcaraz", 0),
            )
        with player_columns[1]:
            p2 = st.selectbox(
                "Player 2",
                names,
                index=_default_player_index(options, "Jannik Sinner", 1),
            )

        context_columns = st.columns(3)
        with context_columns[0]:
            surface = st.selectbox("Surface", ["Hard", "Clay", "Grass", "Carpet"])
        with context_columns[1]:
            tier = st.selectbox(
                "Tournament tier",
                ["Slam", "Masters", "500", "250", "Challenger", "ITF", "Team"],
            )
        with context_columns[2]:
            minimum_date = (
                as_of.date() if as_of is not None else date.today() - timedelta(days=1)
            )
            match_date = st.date_input(
                "Match date",
                value=max(date.today(), minimum_date),
                min_value=minimum_date,
            )

        with st.expander("Optional context flags"):
            flag_columns = st.columns(3)
            with flag_columns[0]:
                dead_rubber = st.selectbox(
                    "Dead rubber status", ["Unknown", "No", "Yes"]
                )
            with flag_columns[1]:
                already_qualified = st.selectbox(
                    "Already qualified", ["Unknown", "No", "Yes"]
                )
            with flag_columns[2]:
                large_line_move = st.checkbox("Large late line move")

        submitted = st.form_submit_button(
            "Generate prediction",
            type="primary",
            width="stretch",
        )

    if submitted:
        if p1 == p2:
            st.error("Player 1 and Player 2 must be different.")
            st.session_state.pop("current-manual-prediction", None)
        else:
            try:
                result = system.predict_match(
                    player1=p1,
                    player2=p2,
                    surface=surface,
                    tournament_tier=tier,
                    match_date=match_date,
                    tour=tour,
                    dead_rubber=_context_value(dead_rubber),
                    already_qualified=_context_value(already_qualified),
                    large_line_move=large_line_move,
                )
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.session_state.pop("current-manual-prediction", None)
            else:
                st.session_state["current-manual-prediction"] = {
                    "result": result,
                    "context": {
                        "tour": tour,
                        "surface": surface,
                        "tier": tier,
                        "match_date": match_date,
                    },
                }

    saved_prediction = st.session_state.get("current-manual-prediction")
    if not saved_prediction:
        st.info(
            "Select two players and generate a prediction. Add current Polymarket asks afterward to receive a price-based betting decision."
        )
        return

    result = saved_prediction["result"]
    prediction_context = saved_prediction["context"]
    sample = result.get("sample_quality") or {}
    sample_status = _render_sample_quality_panel(
        sample if isinstance(sample, dict) else {},
        surface=str(prediction_context["surface"]),
    )
    if sample_status == "INSUFFICIENT":
        _render_cached_alternatives()
        return
    p = float(result["player1_win_probability"])
    player1 = str(result["player1"])
    player2 = str(result["player2"])
    favorite = player1 if p >= 0.5 else player2
    favorite_probability = max(p, 1.0 - p)

    st.markdown(
        f'<div class="prediction-headline"><div class="prediction-kicker">MODEL LEAN</div>'
        f'<div class="prediction-favorite">{favorite}</div>'
        f'<div class="prediction-confidence">{favorite_probability:.1%} estimated win probability</div>'
        f'<div class="prediction-warning">This is not a bet recommendation until market prices are entered.</div></div>',
        unsafe_allow_html=True,
    )
    probability_matchup(player1, player2, p)

    disagreement = abs(p - float(result["elo_probability"]))
    render_metric_row(
        [
            ("Calibrated model", format_percent(p), f"Probability for {player1}"),
            (
                "Elo baseline",
                format_percent(result["elo_probability"]),
                "Age-adjusted Elo",
            ),
            ("Model disagreement", format_percent(disagreement), "Inspect when large"),
        ]
    )

    with st.expander("Model details and player comparison", expanded=False):
        st.markdown("#### Active match flags")
        badges = []
        for label, active in result["filter_flags"].items():
            unknown = bool(result.get("unknown_context", {}).get(label, False))
            badges.append(status_badge(label.replace("_", " ").title(), active, unknown))
        st.markdown(" ".join(badges), unsafe_allow_html=True)

        snapshot = result.get("feature_snapshot", {})
        if snapshot:
            comparison = pd.DataFrame(
                [
                    ["Blended Elo", snapshot.get("elo_blended1"), snapshot.get("elo_blended2")],
                    ["Overall Elo", snapshot.get("elo_overall1"), snapshot.get("elo_overall2")],
                    ["Age", snapshot.get("age1"), snapshot.get("age2")],
                    ["All recorded matches", snapshot.get("total_matches1"), snapshot.get("total_matches2")],
                    ["Tour-level matches", snapshot.get("tour_matches1"), snapshot.get("tour_matches2")],
                    [f"{prediction_context['surface']} matches", snapshot.get("surface_matches1"), snapshot.get("surface_matches2")],
                    ["Last recorded match", snapshot.get("last_match_date1"), snapshot.get("last_match_date2")],
                    ["Years on tour", snapshot.get("years_on_tour1"), snapshot.get("years_on_tour2")],
                    ["Recent form (10)", snapshot.get("form10_1"), snapshot.get("form10_2")],
                    ["Recent form (20)", snapshot.get("form20_1"), snapshot.get("form20_2")],
                    ["Rest days", snapshot.get("rest_days1"), snapshot.get("rest_days2")],
                    [
                        "Injury incidents (12m)",
                        snapshot.get("injury_incidents_12m1"),
                        snapshot.get("injury_incidents_12m2"),
                    ],
                    [
                        "Matches in fatigue window",
                        snapshot.get("matches_prev_window1"),
                        snapshot.get("matches_prev_window2"),
                    ],
                ],
                columns=["Feature", player1, player2],
            )
            st.dataframe(comparison, hide_index=True, width="stretch")

    _render_predictor_betting_panel(
        result=result,
        prediction_context=prediction_context,
        allow_betting=sample_status == "SUFFICIENT",
    )
    if sample_status != "SUFFICIENT":
        _render_cached_alternatives()


def _calibration_frame(frame: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    clean = frame.dropna(subset=["probability", "actual"]).copy()
    clean["bin"] = pd.cut(clean["probability"], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    result = clean.groupby("bin", observed=True).agg(
        mean_probability=("probability", "mean"),
        actual_rate=("actual", "mean"),
        matches=("actual", "size"),
    ).reset_index(drop=True)
    return result


def render_backtest_lab() -> None:
    hero(
        "Backtest Lab",
        "Compare models, inspect calibration, and evaluate where the system performs well or degrades.",
        eyebrow="WALK-FORWARD RESEARCH",
    )
    paths = _paths()
    metrics, metrics_error = safe_load(paths.metrics, load_metrics)
    predictions, predictions_error = safe_load(paths.predictions, load_predictions)
    if metrics_error or predictions_error:
        _empty_state("Backtest artifacts are incomplete.", metrics_error or predictions_error or "Unknown error")
        return

    controls = st.columns(4)
    with controls[0]:
        model = st.selectbox("Model", sorted(predictions["model"].dropna().unique()), index=0)
    with controls[1]:
        filter_view = st.selectbox("Filter view", ["all", "filtered"], format_func=lambda value: value.title())
    with controls[2]:
        tour = st.selectbox("Tour", ["All"] + sorted(predictions["tour"].dropna().unique().tolist()))
    with controls[3]:
        tier = st.selectbox("Tier", ["All"] + sorted(predictions["tier"].dropna().unique().tolist()))

    frame = predictions[predictions["model"] == model].copy()
    if filter_view == "filtered":
        frame = frame[frame["eligible_with_filters"]]
    if tour != "All":
        frame = frame[frame["tour"] == tour]
    if tier != "All":
        frame = frame[frame["tier"] == tier]

    if frame.empty:
        st.warning("No predictions match the selected segment.")
        return

    actual = frame["actual"].astype(int)
    probability = frame["probability"].clip(1e-6, 1 - 1e-6)
    accuracy = ((probability >= 0.5).astype(int) == actual).mean()
    brier = np.mean((probability - actual) ** 2)
    logloss = -np.mean(actual * np.log(probability) + (1 - actual) * np.log(1 - probability))
    render_metric_row(
        [
            ("Matches", f"{len(frame):,}", f"{tour} · {tier}"),
            ("Accuracy", format_percent(accuracy), model),
            ("Brier", format_number(brier, 4), "Lower is better"),
            ("Log loss", format_number(logloss, 4), "Lower is better"),
        ]
    )

    calibration_column, timeline_column = st.columns(2)
    with calibration_column:
        st.markdown("### Calibration curve")
        calibration = _calibration_frame(frame)
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=calibration["mean_probability"],
                y=calibration["actual_rate"],
                mode="lines+markers",
                name=model,
                marker={"size": np.clip(calibration["matches"] / 40, 7, 22), "color": "#b7ff3c"},
                line={"color": "#b7ff3c", "width": 3},
                customdata=calibration["matches"],
                hovertemplate="Predicted %{x:.1%}<br>Actual %{y:.1%}<br>Matches %{customdata}<extra></extra>",
            )
        )
        figure.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect", line={"dash": "dash", "color": "#6b7480"}))
        figure.update_xaxes(tickformat=".0%", range=[0, 1], title="Mean predicted probability")
        figure.update_yaxes(tickformat=".0%", range=[0, 1], title="Observed win rate")
        figure.update_layout(height=430, legend_title_text="", margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(figure, width="stretch")
    with timeline_column:
        st.markdown("### Performance over time")
        grouped = []
        for quarter, group in frame.groupby(frame["date"].dt.to_period("Q")):
            grouped.append(
                {
                    "Quarter": str(quarter),
                    "Accuracy": ((group["probability"] >= 0.5).astype(int) == group["actual"].astype(int)).mean(),
                    "Brier": np.mean((group["probability"] - group["actual"]) ** 2),
                    "Matches": len(group),
                }
            )
        timeline = pd.DataFrame(grouped)
        figure = px.line(timeline, x="Quarter", y="Accuracy", markers=True, hover_data=["Matches", "Brier"])
        figure.update_traces(line_color="#46d7ff", marker_color="#b7ff3c", line_width=3)
        figure.update_yaxes(tickformat=".0%", range=[max(0.45, timeline["Accuracy"].min() - 0.04), min(0.85, timeline["Accuracy"].max() + 0.04)])
        figure.update_layout(height=430, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(figure, width="stretch")

    st.markdown("### Segment table")
    segment_metrics = metrics[(metrics["filter_view"] == filter_view)].copy()
    segment_metrics = segment_metrics[segment_metrics["segment"].str.startswith(model)]
    display_columns = [
        "segment_type", "segment", "matches", "accuracy", "brier", "log_loss",
        "bets", "flat_roi", "flat_roi_ci_low", "flat_roi_ci_high", "kelly_roi",
    ]
    available = [column for column in display_columns if column in segment_metrics.columns]
    st.dataframe(
        segment_metrics[available].sort_values(["segment_type", "matches"], ascending=[True, False]),
        hide_index=True,
        width="stretch",
        column_config={
            "accuracy": st.column_config.NumberColumn(format="percent"),
            "flat_roi": st.column_config.NumberColumn(format="percent"),
            "kelly_roi": st.column_config.NumberColumn(format="percent"),
        },
    )


def render_match_explorer() -> None:
    hero(
        "Match Explorer",
        "Search every walk-forward prediction, isolate errors, and export the exact rows you want to investigate.",
        eyebrow="HISTORICAL PREDICTIONS",
    )
    paths = _paths()
    predictions, error = safe_load(paths.predictions, load_predictions)
    if error:
        _empty_state("Prediction history is unavailable.", error)
        return

    frame = predictions.copy()
    frame["predicted_player"] = np.where(frame["probability"] >= 0.5, frame["player1"], frame["player2"])
    frame["winner"] = np.where(frame["actual"] == 1, frame["player1"], frame["player2"])
    frame["correct"] = frame["predicted_player"] == frame["winner"]
    frame["confidence"] = np.maximum(frame["probability"], 1 - frame["probability"])
    frame["match"] = frame["player1"] + " vs " + frame["player2"]

    first_row = st.columns(4)
    model = first_row[0].multiselect("Model", sorted(frame["model"].unique()), default=sorted(frame["model"].unique()))
    tour = first_row[1].multiselect("Tour", sorted(frame["tour"].dropna().unique()), default=sorted(frame["tour"].dropna().unique()))
    tier = first_row[2].multiselect("Tier", sorted(frame["tier"].dropna().unique()), default=sorted(frame["tier"].dropna().unique()))
    outcome = first_row[3].selectbox("Result", ["All", "Correct", "Incorrect"])

    second_row = st.columns([1.5, 1, 1.2])
    search = second_row[0].text_input("Player or tournament search")
    confidence = second_row[1].slider("Minimum confidence", 0.50, 0.95, 0.50, 0.01)
    eligible_only = second_row[2].toggle("Eligible after filters", value=False)

    filtered = frame[
        frame["model"].isin(model)
        & frame["tour"].isin(tour)
        & frame["tier"].isin(tier)
        & (frame["confidence"] >= confidence)
    ].copy()
    if outcome == "Correct":
        filtered = filtered[filtered["correct"]]
    elif outcome == "Incorrect":
        filtered = filtered[~filtered["correct"]]
    if eligible_only:
        filtered = filtered[filtered["eligible_with_filters"]]
    if search:
        query = search.casefold()
        mask = (
            filtered["player1"].str.casefold().str.contains(query, na=False)
            | filtered["player2"].str.casefold().str.contains(query, na=False)
            | filtered["tournament"].str.casefold().str.contains(query, na=False)
        )
        filtered = filtered[mask]

    render_metric_row(
        [
            ("Matching rows", f"{len(filtered):,}", "Current filters"),
            ("Accuracy", format_percent(filtered["correct"].mean() if len(filtered) else np.nan), "Selected rows"),
            ("Average confidence", format_percent(filtered["confidence"].mean() if len(filtered) else np.nan), "Winning side probability"),
        ]
    )

    columns = [
        "date", "tour", "tier", "tournament", "round", "match", "model",
        "probability", "predicted_player", "winner", "correct", "confidence",
        "any_filter_flag", "retirement", "walkover",
    ]
    display = filtered.reindex(columns=columns).sort_values("date", ascending=False)
    st.dataframe(
        display.head(1000),
        hide_index=True,
        width="stretch",
        height=620,
        column_config={
            "date": st.column_config.DateColumn(format="YYYY-MM-DD"),
            "probability": st.column_config.NumberColumn("Player 1 probability", format="percent"),
            "confidence": st.column_config.ProgressColumn(format="percent", min_value=0.5, max_value=1.0),
            "correct": st.column_config.CheckboxColumn(),
        },
    )
    st.caption("The table displays at most 1,000 rows, but the download includes every matching row.")
    st.download_button(
        "Download filtered CSV",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="tennis_prediction_rows.csv",
        mime="text/csv",
    )


def render_model_guide() -> None:
    hero(
        "Model Guide",
        "A practical reference for interpreting probabilities, filters, and backtest metrics without treating the output as certainty.",
        eyebrow="HOW THE SYSTEM WORKS",
    )

    tab1, tab2, tab3, tab4 = st.tabs(["Pipeline", "Features", "Metrics", "Deployment"])
    with tab1:
        st.markdown(
            """
            ### Leakage-safe sequence
            1. Load and normalize ATP/WTA matches.
            2. Build every feature from state that existed **before** each match.
            3. Update Elo, form, experience, injury proxies, and head-to-head only after recording the pre-match row.
            4. Train on earlier dates and test on later dates using walk-forward folds.
            5. Calibrate boosted probabilities on a chronological holdout.

            The terminal pipeline remains the source of truth. This dashboard only loads its saved artifacts and presents them interactively.
            """
        )
    with tab2:
        feature_table = pd.DataFrame(
            [
                ["Surface-blended Elo", "Surface rating shrunk toward overall Elo when surface history is limited."],
                ["Empirical age curve", "Age contribution learned from historical outcomes inside the training period."],
                ["Recent form", "Recency-decayed win rate over the previous 10 and 20 matches."],
                ["Experience", "Tour-level matches and elapsed years since the player's first recorded match."],
                ["Injury proxy", "Trailing 12-month retirement and walkover incidents."],
                ["Head-to-head", "Only activated after the configured minimum number of prior meetings."],
                ["Rest and fatigue", "Days since last match and match count inside the fatigue window."],
                ["Context flags", "Dead rubber, qualification status, pre-Slam rest risk, and large line movement."],
            ],
            columns=["Feature", "Interpretation"],
        )
        st.dataframe(feature_table, hide_index=True, width="stretch")
    with tab3:
        st.markdown(
            """
            - **Accuracy:** How often the side above 50% won. Useful, but blind to confidence quality.
            - **Brier score:** Mean squared probability error. Lower is better.
            - **Log loss:** Penalizes confident wrong predictions heavily. Lower is better.
            - **ROI:** Requires historical market odds and a declared staking rule.
            - **CLV:** Compares your entry price with the closing market. It is unavailable without closing snapshots.
            - **Bootstrap interval:** Resamples bets to estimate how much ROI could vary from sampling luck.
            """
        )
        st.info("A model can have higher accuracy and worse log loss. That means it selected more winners but assigned probabilities less safely.")
    with tab4:
        st.markdown(
            """
            ### Local dashboard
            ```bash
            streamlit run streamlit_app.py
            ```

            ### Prepare shareable artifacts
            ```bash
            tennis-prepare-dashboard \\
              --source outputs/backtest \\
              --destination dashboard_artifacts
            ```

            Commit `dashboard_artifacts`, `streamlit_app.py`, `.streamlit/config.toml`, and the application code. Raw match files and the full `outputs/` directory remain ignored.

            Before making the repository public, review the underlying tennis-data license and confirm the generated model artifact is below GitHub's file-size limit.
            """
        )



def render_paper_metric_banner() -> None:
    """Global three-card paper account banner."""
    config = load_app_config()
    initialize_paper_trading_db(config.paper_trading.database_path)
    summary = get_wallet_summary(config.paper_trading.database_path)
    record = f"{summary.wins} - {summary.losses} ({summary.win_rate:.1%})"
    columns = st.columns(3)
    columns[0].metric(
        "Current Paper Bankroll",
        f"${summary.balance:,.2f}",
        delta=f"${summary.bankroll_delta:+.2f} vs $100.00",
    )
    columns[1].metric(
        "Our Win/Loss Record",
        record,
        delta=f"{summary.pending} pending · {summary.voids} void",
        delta_color="off",
    )
    columns[2].metric(
        "Our Total Paper ROI",
        f"{summary.roi:.2%}",
        delta=f"${summary.settled_profit:+.2f} settled profit",
    )
    st.write("")


def _odds_comparison_style(frame: pd.DataFrame):
    def color_column(series: pd.Series) -> list[str]:
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() < 2:
            return [""] * len(series)
        maximum = numeric.max()
        minimum = numeric.min()
        return [
            "background-color: rgba(183,255,60,.24); color: #efffd5" if value == maximum
            else "background-color: rgba(255,107,107,.22); color: #ffdada" if value == minimum
            else ""
            for value in numeric
        ]

    return frame.style.apply(color_column, subset=[column for column in frame.columns if column.startswith("Odds")])




def _paper_recommendation(
    *,
    edge: float | None,
    threshold: float,
    model_as_of: object,
    commence_time: object,
    p_model: float,
    p_elo: float,
    tour_matches1: object,
    tour_matches2: object,
    manual_one_sided: bool = False,
) -> tuple[str, str]:
    """Return a conservative research label and explanation, never a real-bet command."""
    reasons: list[str] = []
    as_of = pd.to_datetime(model_as_of, errors="coerce")
    commence = pd.to_datetime(commence_time, errors="coerce")
    if pd.notna(as_of) and pd.notna(commence):
        stale_days = int((commence.normalize() - as_of.normalize()).days)
        if stale_days > 3:
            reasons.append(f"model state is {stale_days} days behind the fixture")
    history_values = pd.to_numeric(
        pd.Series([tour_matches1, tour_matches2]), errors="coerce"
    )
    if history_values.notna().any() and history_values.min() < 10:
        reasons.append("at least one player has fewer than 10 recorded tour matches")
    if np.isfinite(float(p_elo)) and abs(float(p_model) - float(p_elo)) >= 0.15:
        reasons.append("LightGBM and Elo disagree by at least 15 percentage points")
    if reasons:
        return "DATA WARNING — PASS", "; ".join(reasons)
    if edge is None or not np.isfinite(float(edge)):
        return "NO PRICE — CANNOT EVALUATE", "A favorite probability alone is not a betting edge."
    edge_value = float(edge)
    qualifier = "raw one-sided edge" if manual_one_sided else "de-vigged edge"
    if edge_value >= max(0.05, threshold):
        return "PAPER-BET CANDIDATE", f"{qualifier} is {edge_value:+.1%}; track it forward before risking money."
    if edge_value >= threshold:
        return "WATCH", f"{qualifier} is {edge_value:+.1%}, but the margin is modest."
    if edge_value > 0:
        return "PASS — EDGE TOO SMALL", f"{qualifier} is only {edge_value:+.1%}."
    return "PASS — NEGATIVE EDGE", f"{qualifier} is {edge_value:+.1%}."



def _render_unresolved_panel(
    unresolved: pd.DataFrame,
    *,
    expanded: bool = False,
) -> None:
    if unresolved is None or unresolved.empty:
        return
    clean = clean_unresolved_fixtures(unresolved, limit=None)
    if clean.empty:
        return

    reason_counts = (
        clean["reason"]
        .fillna("unresolved")
        .value_counts()
        .rename_axis("Reason")
        .reset_index(name="Fixtures")
    )
    unknown_surface = int(clean["reason"].eq("unknown surface").sum())
    render_metric_row(
        [
            ("Fixtures withheld", f"{len(clean):,}", "Not used for predictions"),
            ("Unknown surface", f"{unknown_surface:,}", "Surface was not guessed"),
            ("Other issues", f"{len(clean) - unknown_surface:,}", "Missing or invalid metadata"),
        ]
    )

    with st.expander("Review fixtures that were safely withheld", expanded=expanded):
        st.caption(
            "These rows are an audit queue, not prediction results. Nested API arrays, "
            "logos, point-by-point objects, and other raw payload fields are hidden."
        )
        chart_column, table_column = st.columns([0.8, 1.8])
        with chart_column:
            st.markdown("#### Exclusion reasons")
            st.dataframe(
                reason_counts,
                hide_index=True,
                width="stretch",
                column_config={
                    "Fixtures": st.column_config.NumberColumn(format="%d"),
                },
            )
        with table_column:
            reasons = ["All reasons"] + reason_counts["Reason"].astype(str).tolist()
            reason = st.selectbox(
                "Filter exclusion reason",
                reasons,
                key=f"unresolved-reason:{expanded}:{len(clean)}",
            )
            visible = clean if reason == "All reasons" else clean[clean["reason"].eq(reason)]
            st.dataframe(
                visible.head(250),
                hide_index=True,
                width="stretch",
                column_config={
                    "event_date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                    "event_time": st.column_config.TextColumn("Time", width="small"),
                    "matchup": st.column_config.TextColumn("Matchup", width="large"),
                    "tournament": st.column_config.TextColumn("Tournament", width="medium"),
                    "tour": st.column_config.TextColumn("Tour", width="small"),
                    "tier": st.column_config.TextColumn("Tier", width="small"),
                    "status": st.column_config.TextColumn("Status", width="small"),
                    "reason": st.column_config.TextColumn("Why withheld", width="medium"),
                },
            )
            if len(visible) > 250:
                st.caption(f"Showing the first 250 of {len(visible):,} filtered rows.")


def _money(value: float) -> str:
    return f"${float(value):,.2f}"


def _price_label(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "—" if pd.isna(numeric) else f"{float(numeric) * 100:.1f}¢"


def _status_badge(label: str) -> str:
    css = "status-ready"
    if label == "PAPER-BET CANDIDATE":
        css = "status-candidate"
    elif label == "PASS":
        css = "status-pass"
    elif label != "READY":
        css = "status-warning"
    return f'<span class="{css}">{label}</span>'


def _local_player_choices(system, tour: str) -> list[tuple[str, str]]:
    resolver = system.name_resolver
    if resolver is None:
        return []
    identities = list(resolver.by_id.values())
    if tour:
        identities = [identity for identity in identities if str(identity.tour).upper() == tour.upper()]
    identities.sort(key=lambda identity: identity.name.casefold())
    return [(identity.name, identity.player_id) for identity in identities]



def _render_market_only_moneyline(row: dict[str, object]) -> None:
    """Render executable market prices without pretending they are model output."""
    player1 = str(row.get("player1") or "Player A")
    player2 = str(row.get("player2") or "Player B")
    ask1 = pd.to_numeric(pd.Series([row.get("best_ask1")]), errors="coerce").iloc[0]
    ask2 = pd.to_numeric(pd.Series([row.get("best_ask2")]), errors="coerce").iloc[0]
    bid1 = pd.to_numeric(pd.Series([row.get("best_bid1")]), errors="coerce").iloc[0]
    bid2 = pd.to_numeric(pd.Series([row.get("best_bid2")]), errors="coerce").iloc[0]

    columns = st.columns(2)
    for column, player, ask, bid in (
        (columns[0], player1, ask1, bid1),
        (columns[1], player2, ask2, bid2),
    ):
        with column:
            st.markdown(f"#### {player}")
            if pd.notna(ask) and 0 < float(ask) < 1:
                ask = float(ask)
                st.metric("Executable moneyline ask", f"{ask * 100:.1f}¢")
                st.caption(f"Decimal odds: {1.0 / ask:.3f}")
            else:
                st.metric("Executable moneyline ask", "—")
                st.caption("No current ask is available.")
            if pd.notna(bid) and 0 < float(bid) < 1:
                st.caption(f"Best bid: {float(bid) * 100:.1f}¢")

    st.info(
        "Market prices are shown because this is a genuine match moneyline. "
        "No model probability or paper-bet recommendation is issued for this card."
    )
    with st.expander("Order-book details"):
        book = pd.DataFrame(
            [
                {
                    "Outcome": player1,
                    "Best bid": _price_label(row.get("best_bid1")),
                    "Best ask": _price_label(row.get("best_ask1")),
                    "Spread": _price_label(row.get("spread1")),
                    "Visible ask liquidity": _money(float(row.get("liquidity1") or 0)),
                },
                {
                    "Outcome": player2,
                    "Best bid": _price_label(row.get("best_bid2")),
                    "Best ask": _price_label(row.get("best_ask2")),
                    "Spread": _price_label(row.get("spread2")),
                    "Visible ask liquidity": _money(float(row.get("liquidity2") or 0)),
                },
            ]
        )
        st.dataframe(book, hide_index=True, width="stretch")
        st.caption(
            f"Market ID: {row.get('market_id') or 'Unknown'} · "
            f"Question: {row.get('question') or 'Unknown'}"
        )

def _refresh_polymarket_order_books(board):
    """Refresh only CLOB books for markets already on the selected board."""
    if board is None or board.markets is None or board.markets.empty:
        return board

    previous = board.markets.copy()
    refreshed = PolymarketClient().enrich_order_books(previous)
    refreshed["price_snapshot_time"] = pd.Timestamp.now(tz="UTC").tz_convert(None)

    board.markets = refreshed
    board.matches = refreshed
    board.quotes = refreshed
    return board


def render_today_matches() -> None:
    hero(
        "Polymarket Scanner",
        "Only genuine match-winner moneylines are shown. Live and unsupported markets remain visible as market-only cards; model recommendations stay pre-match and in-scope.",
        eyebrow="MONEYLINE-ONLY MARKET BOARD",
    )
    config = load_app_config()
    paths = _paths()
    system, error = safe_load(paths.model, load_system)
    if error:
        _empty_state("A trained model is required.", error)
        return

    startup_error = st.session_state.get("startup_settlement_error")
    if startup_error:
        st.caption(f"Automatic settlement warning: {startup_error}")

    controls = st.columns([1, 1.35, 1, 1])
    with controls[0]:
        board_date = st.date_input(
            "Match date",
            value=pd.Timestamp.now(tz=config.live_data.timezone).date(),
            key="polymarket-board-date",
        )
    with controls[1]:
        search = st.text_input(
            "Search tournament or player", placeholder="ITF, Challenger, Lincoln, player name…"
        )
    with controls[2]:
        status_filter = st.multiselect(
            "Model status",
            [
                "READY",
                "LIVE / MARKET ONLY",
                "UNSUPPORTED SEGMENT",
                "NEEDS MAPPING",
                "NEEDS SURFACE",
                "COLD START / NO HISTORY",
                "LIMITED SAMPLE / NO BET",
                "MODEL ERROR",
            ],
            default=[],
            key="polymarket-status-filter-v2",
            help="Leave blank to show every tennis market, including unmapped and cold-start players.",
        )
    with controls[3]:
        refresh = st.button("Refresh selected day", type="primary", width="stretch")

    live_price_controls = st.columns([1, 1, 2])
    with live_price_controls[0]:
        live_prices = st.toggle(
            "Live prices",
            value=True,
            key=f"polymarket-live-prices:{board_date.isoformat()}",
            help="Refreshes current CLOB bids and asks without repeating market discovery.",
        )
    with live_price_controls[1]:
        price_refresh_seconds = st.selectbox(
            "Price refresh",
            [5, 10, 15, 30],
            index=1,
            format_func=lambda value: f"Every {value} seconds",
            key=f"polymarket-price-seconds:{board_date.isoformat()}",
            disabled=not live_prices,
        )
    with live_price_controls[2]:
        st.caption(
            "Refresh selected day discovers new markets. Live prices updates only the order books already on this board."
        )

    board_key = f"polymarket-first-board:{board_date.isoformat()}"
    if refresh:
        with st.spinner(f"Loading a bounded Polymarket board for {board_date:%B %d, %Y}…"):
            try:
                refreshed_board = build_live_board(
                    config, system=system, board_date=board_date
                )
            except Exception as exc:
                st.error(f"Polymarket refresh failed: {exc}")
                return
            previous_board = st.session_state.get(board_key)
            if (
                refreshed_board.markets.empty
                and refreshed_board.errors
                and previous_board is not None
            ):
                st.warning("The live refresh failed, so the last successful board is being kept.")
            else:
                st.session_state[board_key] = refreshed_board
    board = st.session_state.get(board_key)
    if board is None:
        st.info("Choose a date and press Refresh selected day. The scanner no longer starts a large network crawl automatically.")
        return

    if live_prices and hasattr(st, "fragment"):
        tick_key = f"polymarket-price-last:{board_date.isoformat()}"
        error_key = f"polymarket-price-error:{board_date.isoformat()}"

        @st.fragment(run_every=f"{int(price_refresh_seconds)}s")
        def _live_price_tick() -> None:
            now_monotonic = time.monotonic()
            last_tick = float(st.session_state.get(tick_key, 0.0) or 0.0)
            if now_monotonic - last_tick < max(int(price_refresh_seconds) - 1, 1):
                return

            current_board = st.session_state.get(board_key)
            if current_board is None or current_board.markets.empty:
                return

            try:
                st.session_state[board_key] = _refresh_polymarket_order_books(current_board)
                st.session_state.pop(error_key, None)
            except Exception as exc:
                st.session_state[error_key] = str(exc)
                return

            st.session_state[tick_key] = now_monotonic
            st.rerun()

        _live_price_tick()

        price_error = st.session_state.get(error_key)
        if price_error:
            st.caption(f"Live price refresh warning: {price_error}")

    for message in board.errors:
        st.warning(message)

    markets = board.markets.copy()
    diagnostics = board.diagnostics or {}

    if live_prices:
        snapshots = pd.to_datetime(
            markets.get("price_snapshot_time", pd.Series(pd.NaT, index=markets.index)),
            errors="coerce",
        )
        latest_snapshot = snapshots.max()
        if pd.notna(latest_snapshot):
            st.caption(
                f"Live order-book prices refresh every {int(price_refresh_seconds)} seconds. "
                f"Last price update: {pd.Timestamp(latest_snapshot).strftime('%I:%M:%S %p')}."
            )
        else:
            st.caption(
                f"Live order-book prices are enabled and will refresh every {int(price_refresh_seconds)} seconds."
            )

    render_metric_row(
        [
            ("Genuine moneylines", f"{len(markets):,}", "Props and duplicate submarkets removed"),
            ("Live moneylines", str(diagnostics.get("live_moneylines", 0)), "Shown as market-only"),
            ("Model eligible", str(diagnostics.get("model_eligible_moneylines", 0)), "Pre-match, supported, sufficient sample"),
            ("Props filtered", str(diagnostics.get("props_filtered", 0)), "Totals, spreads, sets, and other props"),
        ]
    )

    view_mode = st.radio(
        "Moneyline view",
        ["All Moneylines", "Model Eligible"],
        horizontal=True,
        key=f"polymarket-moneyline-view:{board_date.isoformat()}",
        help=(
            "All Moneylines shows every genuine match winner, including live and unsupported events. "
            "Model Eligible shows only pre-match cards that can receive a model-backed paper-bet signal."
        ),
    )
    if view_mode == "Model Eligible":
        markets = markets[
            markets.get("model_status", pd.Series("", index=markets.index)).eq("READY")
            & markets.get("sample_status", pd.Series("", index=markets.index)).eq("SUFFICIENT")
        ].copy()
    if markets.empty:
        st.info(
            f"No active Polymarket tennis markets were found for {board_date:%B %d, %Y} "
            f"in {config.live_data.timezone}. Try another date or refresh later."
        )
        return

    if search.strip():
        query = search.strip().casefold()
        searchable = (
            markets.get("display_title", pd.Series("", index=markets.index)).fillna("").astype(str)
            + " "
            + markets.get("question", pd.Series("", index=markets.index)).fillna("").astype(str)
            + " "
            + markets.get("tournament", pd.Series("", index=markets.index)).fillna("").astype(str)
            + " "
            + markets.get("player1", pd.Series("", index=markets.index)).fillna("").astype(str)
            + " "
            + markets.get("player2", pd.Series("", index=markets.index)).fillna("").astype(str)
        ).str.casefold()
        markets = markets[searchable.str.contains(query, regex=False)]
    if status_filter:
        markets = markets[markets["model_status"].isin(status_filter)]
    quality_filter = st.multiselect(
        "Sample quality",
        ["SUFFICIENT", "LIMITED", "INSUFFICIENT", "UNKNOWN", "MARKET ONLY"],
        default=[],
        key=f"polymarket-quality-filter:{board_date.isoformat()}",
        help="Only SUFFICIENT matchups are eligible for paper-bet signals.",
    )
    if quality_filter:
        markets = markets[
            markets.get("sample_status", pd.Series("UNKNOWN", index=markets.index)).isin(quality_filter)
        ]
    if markets.empty:
        st.info("No active markets match those filters.")
        return

    wallet = get_wallet_summary(config.paper_trading.database_path)
    st.caption(
        f"Pricing uses the current best CLOB ask. Kelly recommendations use a "
        f"{config.paper_trading.kelly_multiplier:.0%} Kelly multiplier and are capped at "
        f"{config.paper_trading.max_kelly_fraction:.1%} of the ${wallet.balance:,.2f} paper bankroll."
    )

    sufficient = markets[
        markets.get("sample_status", pd.Series("", index=markets.index)).eq("SUFFICIENT")
        & markets.get("model_status", pd.Series("", index=markets.index)).eq("READY")
    ].copy()
    if not sufficient.empty:
        sufficient["best_edge"] = sufficient[["edge1", "edge2"]].apply(pd.to_numeric, errors="coerce").max(axis=1)
        alternatives = sufficient.sort_values("best_edge", ascending=False).head(5)
        with st.expander("Best-covered alternatives", expanded=True):
            alt = alternatives[["display_title", "tournament", "surface", "total_matches1", "total_matches2", "best_edge"]].copy()
            alt["best_edge"] = alt["best_edge"].map(lambda value: "—" if pd.isna(value) else f"{float(value):+.1%}")
            alt.columns = ["Match", "Tournament", "Surface", "P1 matches", "P2 matches", "Best edge"]
            st.dataframe(alt, hide_index=True, width="stretch")

    quality_order = {"SUFFICIENT": 0, "LIMITED": 1, "INSUFFICIENT": 2, "MARKET ONLY": 3, "UNKNOWN": 4}
    markets = markets.copy()
    markets["_quality_order"] = markets.get("sample_status", pd.Series("UNKNOWN", index=markets.index)).map(quality_order).fillna(4)
    markets["_best_edge"] = markets[["edge1", "edge2"]].apply(pd.to_numeric, errors="coerce").max(axis=1)
    markets["_phase_order"] = markets.get("market_phase", pd.Series("PREMATCH", index=markets.index)).map({"LIVE": 0, "PREMATCH": 1}).fillna(2)
    markets = markets.sort_values(["_phase_order", "commence_time", "_quality_order", "_best_edge"], ascending=[True, True, True, False])
    cards_per_page = 20
    page_count = max((len(markets) + cards_per_page - 1) // cards_per_page, 1)
    page_number = st.selectbox(
        "Market page",
        list(range(1, page_count + 1)),
        format_func=lambda value: f"Page {value} of {page_count}",
        key=f"polymarket-card-page:{board_date.isoformat()}",
    )
    start_index = (int(page_number) - 1) * cards_per_page
    visible_markets = markets.iloc[start_index : start_index + cards_per_page]
    st.caption(f"Showing {start_index + 1}-{min(start_index + cards_per_page, len(markets))} of {len(markets)} markets. Only twenty cards are rendered at once to protect browser memory.")

    for row in visible_markets.to_dict("records"):
        market_id = str(row.get("market_id") or "")
        title = str(row.get("display_title") or row.get("question") or "Tennis market")
        tournament = str(row.get("tournament") or "Polymarket Tennis")
        commence = pd.to_datetime(row.get("commence_time"), errors="coerce")
        time_label = (
            pd.Timestamp(commence).strftime("%a %b %d · %I:%M %p")
            if pd.notna(commence)
            else "Start time unavailable"
        )
        status = str(row.get("model_status") or "NEEDS MAPPING")
        with st.container(border=True):
            header_left, header_right = st.columns([4, 1])
            with header_left:
                st.markdown(f"### {title}")
                st.caption(
                    f"{time_label} · {tournament} · {row.get('tour') or 'Tour unknown'} · "
                    f"{row.get('tier') or 'Tier unknown'} · {row.get('surface') or 'Unknown surface'} · "
                    f"{row.get('court') or 'Unknown court'}"
                )
                if row.get("surface_evidence"):
                    confidence = str(row.get("surface_confidence") or "").strip()
                    evidence = str(row.get("surface_evidence") or "").replace("_", " ")
                    st.caption(
                        "Court source: " + evidence
                        + (f" · {confidence}" if confidence else "")
                    )
            with header_right:
                st.markdown(_status_badge(status), unsafe_allow_html=True)
                if row.get("market_url"):
                    st.link_button("Open Polymarket", str(row["market_url"]), width="stretch")
            st.caption(str(row.get("model_reason") or row.get("parse_reason") or ""))
            sample_status = str(row.get("sample_status") or "UNKNOWN")
            st.markdown(_status_badge(f"{sample_status} SAMPLE"), unsafe_allow_html=True)
            if sample_status not in {"UNKNOWN", "MARKET ONLY"}:
                sample_rows = pd.DataFrame(
                    [
                        {
                            "Player": row.get("mapped_player1") or row.get("player1") or "Player A",
                            "Recorded": int(row.get("total_matches1") or 0),
                            f"{row.get('surface') or 'Surface'} matches": int(row.get("surface_matches1") or 0),
                            "Last match": row.get("last_match_date1") or "Unknown",
                        },
                        {
                            "Player": row.get("mapped_player2") or row.get("player2") or "Player B",
                            "Recorded": int(row.get("total_matches2") or 0),
                            f"{row.get('surface') or 'Surface'} matches": int(row.get("surface_matches2") or 0),
                            "Last match": row.get("last_match_date2") or "Unknown",
                        },
                    ]
                )
                st.dataframe(sample_rows, hide_index=True, width="stretch")

            if status in {"LIVE / MARKET ONLY", "UNSUPPORTED SEGMENT"}:
                _render_market_only_moneyline(row)
                continue

            if status == "NEEDS MAPPING":
                choices = _local_player_choices(system, str(row.get("tour") or ""))
                if not choices:
                    choices = _local_player_choices(system, "")
                choice_names = [name for name, _ in choices]
                if not choice_names:
                    st.error("No local players are available for mapping.")
                    continue
                with st.expander("Map Polymarket names to local players", expanded=False):
                    st.caption("Saving an alias does not alter historical match results or model weights.")
                    mapping_columns = st.columns(2)
                    provider1 = mapping_columns[0].text_input(
                        "Polymarket Player A label",
                        value=str(row.get("player1") or ""),
                        key=f"provider1:{market_id}",
                    )
                    local1 = mapping_columns[0].selectbox(
                        "Local Player A", choice_names, key=f"local1:{market_id}"
                    )
                    provider2 = mapping_columns[1].text_input(
                        "Polymarket Player B label",
                        value=str(row.get("player2") or ""),
                        key=f"provider2:{market_id}",
                    )
                    local2 = mapping_columns[1].selectbox(
                        "Local Player B", choice_names, index=min(1, len(choice_names)-1), key=f"local2:{market_id}"
                    )
                    if st.button("Save player mappings", key=f"save-map:{market_id}"):
                        by_name = {name: player_id for name, player_id in choices}
                        if not provider1.strip() or not provider2.strip():
                            st.error("Enter both Polymarket player labels before saving.")
                        elif local1 == local2:
                            st.error("Map the two outcomes to different local players.")
                        else:
                            save_player_alias(
                                config.paper_trading.database_path,
                                provider="polymarket",
                                provider_name=provider1,
                                tour=str(row.get("tour") or ""),
                                local_player_id=by_name[local1],
                                local_player_name=local1,
                            )
                            save_player_alias(
                                config.paper_trading.database_path,
                                provider="polymarket",
                                provider_name=provider2,
                                tour=str(row.get("tour") or ""),
                                local_player_id=by_name[local2],
                                local_player_name=local2,
                            )
                            st.session_state.pop(board_key, None)
                            st.success("Mappings saved. Refreshing the market board.")
                            st.rerun()
                continue

            if status == "NEEDS SURFACE":
                with st.expander("Resolve tournament court", expanded=False):
                    st.caption(
                        "This saves the court for this tournament edition, so every match "
                        "from the same event can reuse it."
                    )
                    columns = st.columns(3)
                    surface = columns[0].selectbox(
                        "Surface", ["Hard", "Clay", "Grass", "Carpet"], key=f"surface:{market_id}"
                    )
                    court = columns[1].selectbox(
                        "Court", ["Outdoor", "Indoor", "Unknown"], key=f"court:{market_id}"
                    )
                    tier_options = ["Slam", "Masters", "500", "250", "Challenger", "ITF", "Other"]
                    inferred = str(row.get("tier") or "Other")
                    tier_index = tier_options.index(inferred) if inferred in tier_options else len(tier_options)-1
                    tier = columns[2].selectbox(
                        "Tier", tier_options, index=tier_index, key=f"tier:{market_id}"
                    )
                    if st.button("Save tournament metadata", key=f"save-surface:{market_id}"):
                        save_tournament_metadata(
                            config.paper_trading.database_path,
                            provider="polymarket",
                            tournament=tournament,
                            tour=str(row.get("tour") or ""),
                            surface=surface,
                            court=court,
                            tier=tier,
                            match_date=row.get("commence_time"),
                            evidence_source="manual_dashboard",
                            confidence="MANUAL",
                        )
                        st.session_state.pop(board_key, None)
                        st.success("Tournament metadata saved.")
                        st.rerun()
                continue

            player1 = str(row.get("mapped_player1") or row.get("player1") or "Player A")
            player2 = str(row.get("mapped_player2") or row.get("player2") or "Player B")
            p1 = pd.to_numeric(pd.Series([row.get("p_model1")]), errors="coerce").iloc[0]
            if status == "COLD START / NO HISTORY" or pd.isna(p1):
                st.warning(
                    f"No prediction is issued. Recorded matches: {player1} "
                    f"{int(row.get('total_matches1') or 0)}, {player2} "
                    f"{int(row.get('total_matches2') or 0)}. Choose a SUFFICIENT SAMPLE matchup instead."
                )
                continue

            sample_can_bet = bool(row.get("sample_can_bet", False))
            if not sample_can_bet:
                st.warning("Research-only probability: sample-size rules block edge, Kelly, and paper betting for this matchup.")
            p1 = float(p1)
            p2 = 1.0 - p1
            ask1 = pd.to_numeric(pd.Series([row.get("best_ask1")]), errors="coerce").iloc[0]
            ask2 = pd.to_numeric(pd.Series([row.get("best_ask2")]), errors="coerce").iloc[0]
            sides = st.columns(2)
            for side, (column, player, opponent, probability, ask, token_id, edge_value) in enumerate(
                [
                    (sides[0], player1, player2, p1, ask1, row.get("token_id1"), row.get("edge1")),
                    (sides[1], player2, player1, p2, ask2, row.get("token_id2"), row.get("edge2")),
                ],
                start=1,
            ):
                with column:
                    st.markdown(f"#### {player}")
                    st.metric("Model probability", f"{probability:.1%}")
                    if pd.isna(ask) or not 0 < float(ask) < 1:
                        st.markdown(_status_badge("PASS"), unsafe_allow_html=True)
                        st.caption("No executable Polymarket ask is currently available.")
                        continue
                    ask = float(ask)
                    decimal_odds = 1.0 / ask
                    recommendation = fractional_kelly(
                        probability,
                        decimal_odds,
                        wallet.balance,
                        multiplier=config.paper_trading.kelly_multiplier,
                        max_fraction=config.paper_trading.max_kelly_fraction,
                    )
                    signal = (
                        betting_signal(recommendation, min_edge=config.paper_trading.min_edge)
                        if sample_can_bet
                        else "PASS"
                    )
                    st.markdown(
                        f'<div class="price-box"><div class="price-title">Executable ask</div>'
                        f'<div class="price-value">{ask*100:.1f}¢ · {decimal_odds:.3f}</div></div>',
                        unsafe_allow_html=True,
                    )
                    st.write(f"**Edge:** {recommendation.edge:+.1%}")
                    st.write(
                        f"**Quarter-Kelly:** {_money(recommendation.stake)} "
                        f"({recommendation.applied_fraction:.1%} of bankroll)"
                    )
                    st.markdown(_status_badge(signal), unsafe_allow_html=True)
                    if signal == "PAPER-BET CANDIDATE":
                        st.caption(
                            f"Bet {_money(recommendation.stake)} on {player} @ {decimal_odds:.3f}."
                        )
                    else:
                        st.caption("Negative/insufficient edge or a zero Kelly recommendation.")

                    bet_key = f"polymarket:{market_id}:{token_id or side}"
                    disabled = (
                        not sample_can_bet
                        or signal != "PAPER-BET CANDIDATE"
                        or recommendation.stake <= 0
                    )
                    if st.button(
                        f"Simulate bet on {player}",
                        key=f"simulate:{bet_key}",
                        disabled=disabled,
                        width="stretch",
                    ):
                        try:
                            place_paper_bet(
                                config.paper_trading.database_path,
                                fixture_id=bet_key,
                                matchup=f"{player1} vs {player2}",
                                bet_on=player,
                                odds=decimal_odds,
                                sportsbook="Polymarket",
                                stake=recommendation.stake,
                                p_model=probability,
                                provider_market_id=market_id,
                                token_id=str(token_id or ""),
                                opponent=opponent,
                                tournament=tournament,
                                commence_time=row.get("commence_time"),
                                edge=recommendation.edge,
                                kelly_fraction=recommendation.applied_fraction,
                            )
                        except Exception as exc:
                            st.error(str(exc))
                        else:
                            st.success(
                                f"Paper bet logged: {_money(recommendation.stake)} on {player}."
                            )
                            st.rerun()

            with st.expander("Order-book details"):
                book = pd.DataFrame(
                    [
                        {
                            "Outcome": player1,
                            "Best bid": _price_label(row.get("best_bid1")),
                            "Best ask": _price_label(row.get("best_ask1")),
                            "Spread": _price_label(row.get("spread1")),
                            "Visible ask liquidity": _money(float(row.get("liquidity1") or 0)),
                        },
                        {
                            "Outcome": player2,
                            "Best bid": _price_label(row.get("best_bid2")),
                            "Best ask": _price_label(row.get("best_ask2")),
                            "Spread": _price_label(row.get("spread2")),
                            "Visible ask liquidity": _money(float(row.get("liquidity2") or 0)),
                        },
                    ]
                )
                st.dataframe(book, hide_index=True, width="stretch")
                st.caption(f"Market ID: {market_id} · Question: {row.get('question')}")


def render_paper_bets() -> None:
    hero(
        "Paper-Bet Tracker",
        "Review open positions and every settled simulated wager against one continuous $100 bankroll.",
        eyebrow="ACCOUNT LEDGER",
    )
    config = load_app_config()
    initialize_paper_trading_db(config.paper_trading.database_path)
    processed = Path(config.data.processed_path)
    if processed.exists():
        try:
            settle_pending_bets_from_results(
                config.paper_trading.database_path,
                pd.read_parquet(processed),
            )
        except Exception as exc:
            st.warning(f"Automatic settlement check could not run: {exc}")
    pending_tab, settled_tab = st.tabs(["Pending", "Settled"])
    with pending_tab:
        pending = list_bets(config.paper_trading.database_path, statuses=("PENDING",))
        if pending.empty:
            st.info("No pending paper bets.")
        else:
            st.dataframe(pending, hide_index=True, width="stretch")
    with settled_tab:
        settled = list_bets(config.paper_trading.database_path, statuses=("WON", "LOST", "VOID"))
        if settled.empty:
            st.info("No settled paper bets yet.")
        else:
            settled["profit"] = np.select(
                [settled["status"].eq("WON"), settled["status"].eq("LOST")],
                [settled["stake"] * (settled["odds"] - 1.0), -settled["stake"]],
                default=0.0,
            )
            st.dataframe(settled, hide_index=True, width="stretch")
