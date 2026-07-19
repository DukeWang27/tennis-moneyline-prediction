from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from tennis_predictor.config import AppConfig
from tennis_predictor.data.api_tennis import (
    ApiTennisClient,
    ProviderPlayerResolver,
    TournamentRegistry,
    normalize_api_tennis_fixtures,
    normalize_api_tennis_settlements,
)
from tennis_predictor.data.names import PlayerIdentity, normalize_name
from tennis_predictor.market import PolymarketClient
from tennis_predictor.pipeline import TennisPredictionSystem
from tennis_predictor.quality import evaluate_match_sample_quality
from tennis_predictor.storage import (
    IdentityStore,
    initialize_live_store,
    initialize_paper_trading_db,
    resolve_player_alias,
    resolve_tournament_metadata,
    settle_pending_bets_from_results,
    upsert_match_history,
    upsert_polymarket_markets,
    upsert_prediction_ledger,
)


@dataclass(slots=True)
class SyncReport:
    fetched: int = 0
    appended: int = 0
    state_rows: int = 0
    unresolved: int = 0
    history_rows: int = 0
    ledger_rows: int = 0
    settled: dict[str, int] = field(
        default_factory=lambda: {"won": 0, "lost": 0, "void": 0}
    )
    start_date: str | None = None
    end_date: str | None = None


@dataclass(slots=True)
class LiveBoard:
    markets: pd.DataFrame
    errors: list[str]
    diagnostics: dict[str, Any]
    # Backward-compatible aliases used by older dashboard code/tests.
    matches: pd.DataFrame = field(default_factory=pd.DataFrame)
    quotes: pd.DataFrame = field(default_factory=pd.DataFrame)
    unresolved: pd.DataFrame = field(default_factory=pd.DataFrame)


def _append_unresolved(path: str | Path, unresolved: pd.DataFrame) -> None:
    if unresolved is None or unresolved.empty:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = pd.read_csv(target, low_memory=False)
        unresolved = pd.concat([existing, unresolved], ignore_index=True, sort=False)
    subset = [column for column in ("event_key", "unresolved_reason") if column in unresolved]
    if subset:
        unresolved = unresolved.drop_duplicates(subset, keep="last")
    unresolved.to_csv(target, index=False)


def _feature_predictions(system: TennisPredictionSystem, generated: pd.DataFrame) -> pd.DataFrame:
    """Score pre-match snapshots created before each sequential state update."""
    if generated is None or generated.empty:
        return pd.DataFrame()
    scored = system.score_feature_rows(generated)
    known_surface = scored["surface"].isin(["Hard", "Clay", "Grass", "Carpet"])
    output = pd.DataFrame(
        {
            "match_id": scored["match_id"].astype(str),
            "predicted_at": pd.to_datetime(scored["date"], errors="coerce"),
            "model_as_of": pd.to_datetime(scored["date"], errors="coerce"),
            "player1": scored["player1"].astype(str),
            "player2": scored["player2"].astype(str),
            "p_model1": pd.to_numeric(scored["p_model1"], errors="coerce"),
            "p_elo1": pd.to_numeric(scored["p_elo1"], errors="coerce"),
            "actual": pd.to_numeric(scored["player1_won"], errors="coerce"),
            "prediction_status": np.where(known_surface, "READY", "NEEDS_SURFACE"),
        }
    )
    # Unknown-surface matches are processed into player state but do not receive a
    # fabricated LightGBM prediction because surface is an input to the fitted model.
    output.loc[~known_surface, ["p_model1", "p_elo1"]] = np.nan
    return output


def _base_history(existing: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    dates = pd.to_datetime(existing["date"], errors="coerce")
    return existing[dates < start].copy()


def backfill_completed_matches(
    config: AppConfig,
    *,
    model_path: str | Path,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    client: ApiTennisClient | None = None,
) -> SyncReport:
    """Append new completed matches in chronological order.

    Unknown-surface matches are retained and update overall Elo, form, rest and
    H2H. The state layer alone decides whether surface Elo is updated.
    """
    config.ensure_directories()
    db_path = config.paper_trading.database_path
    initialize_paper_trading_db(db_path)
    initialize_live_store(db_path)

    processed_path = Path(config.data.processed_path)
    if not processed_path.exists():
        raise FileNotFoundError(f"Historical match file is missing: {processed_path}")
    existing = pd.read_parquet(processed_path)
    upsert_match_history(db_path, existing)

    system = TennisPredictionSystem.load(model_path)
    latest = pd.to_datetime(existing["date"], errors="coerce").max()
    default_start = pd.Timestamp(config.live_data.backfill_start)
    if pd.notna(latest):
        replay_start = latest.normalize() - pd.Timedelta(days=max(config.live_data.replay_days, 0))
        default_start = max(default_start, replay_start)
    start = pd.Timestamp(start_date or default_start).normalize()
    end = pd.Timestamp(end_date or pd.Timestamp.today()).normalize()
    if end < start:
        raise ValueError("end_date must be on or after start_date")

    api = client or ApiTennisClient(api_key_env=config.live_data.api_tennis_key_env)
    raw = api.fetch_fixtures(start, end, timezone_name=config.live_data.timezone)
    registry = TournamentRegistry(config.live_data.tournament_registry_path, existing)
    player_resolver = ProviderPlayerResolver(
        config.live_data.player_aliases_path, system.name_resolver
    )
    settlement_rows = normalize_api_tennis_settlements(
        raw, player_resolver=player_resolver
    )
    normalized, unresolved = normalize_api_tennis_fixtures(
        raw,
        registry=registry,
        player_resolver=player_resolver,
        completed_only=True,
        randomize_completed_sides=True,
    )
    _append_unresolved(config.live_data.unresolved_fixtures_path, unresolved)

    report = SyncReport(
        fetched=len(raw),
        unresolved=len(unresolved),
        start_date=start.date().isoformat(),
        end_date=end.date().isoformat(),
    )
    report.settled = settle_pending_bets_from_results(db_path, settlement_rows)
    if normalized.empty:
        return report

    report.history_rows = upsert_match_history(db_path, normalized)
    existing_ids = set(existing["match_id"].astype(str))
    new_rows = normalized[~normalized["match_id"].astype(str).isin(existing_ids)].copy()
    report.appended = len(new_rows)
    if new_rows.empty:
        return report

    # Update the in-memory model before writing files. The pipeline rejects unseen
    # older rows when the model has already advanced, preventing silent leakage.
    generated = system.update_state(new_rows)
    report.state_rows = len(generated)
    ledger = _feature_predictions(system, generated)
    report.ledger_rows = upsert_prediction_ledger(
        db_path, ledger, source="sequential_live_backfill"
    )

    updated = pd.concat([existing, new_rows], ignore_index=True, sort=False)
    updated = updated.drop_duplicates("match_id", keep="last")
    sort_columns = [
        column
        for column in ("date", "event_date", "match_num", "match_id")
        if column in updated.columns
    ]
    updated = updated.sort_values(sort_columns).reset_index(drop=True)
    updated.to_parquet(processed_path, index=False)
    system.save(model_path)
    return report


def rebuild_live_state(
    config: AppConfig,
    *,
    base_model_path: str | Path,
    live_model_path: str | Path,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp | None = None,
    client: ApiTennisClient | None = None,
) -> SyncReport:
    """Rebuild post-cutoff state from an immutable frozen model.

    Use this after changing ingestion rules for dates that the current live model
    has already passed. It is the safe alternative to applying old unseen matches
    to a future state.
    """
    config.ensure_directories()
    db_path = config.paper_trading.database_path
    initialize_paper_trading_db(db_path)
    initialize_live_store(db_path)

    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date or pd.Timestamp.today()).normalize()
    if end < start:
        raise ValueError("end_date must be on or after start_date")

    system = TennisPredictionSystem.load(base_model_path)
    state_date = system.feature_engineer.state.as_of
    if state_date is not None and pd.Timestamp(state_date).normalize() >= start:
        raise ValueError(
            "The base model must end before the rebuild start date; otherwise "
            "the replay is not leakage-safe."
        )

    processed_path = Path(config.data.processed_path)
    existing = pd.read_parquet(processed_path)
    base_rows = _base_history(existing, start)

    api = client or ApiTennisClient(api_key_env=config.live_data.api_tennis_key_env)
    raw = api.fetch_fixtures(start, end, timezone_name=config.live_data.timezone)
    registry = TournamentRegistry(config.live_data.tournament_registry_path, base_rows)
    player_resolver = ProviderPlayerResolver(
        config.live_data.player_aliases_path, system.name_resolver
    )
    settlements = normalize_api_tennis_settlements(raw, player_resolver=player_resolver)
    normalized, unresolved = normalize_api_tennis_fixtures(
        raw,
        registry=registry,
        player_resolver=player_resolver,
        completed_only=True,
        randomize_completed_sides=True,
    )
    _append_unresolved(config.live_data.unresolved_fixtures_path, unresolved)

    report = SyncReport(
        fetched=len(raw),
        appended=len(normalized),
        unresolved=len(unresolved),
        start_date=start.date().isoformat(),
        end_date=end.date().isoformat(),
    )
    report.settled = settle_pending_bets_from_results(db_path, settlements)
    if normalized.empty:
        system.save(live_model_path)
        return report

    generated = system.update_state(normalized)
    report.state_rows = len(generated)

    # prediction_ledger.match_id has a foreign key to match_history.match_id.
    # Insert the completed matches first so every ledger row has a parent.
    report.history_rows = upsert_match_history(db_path, normalized)
    ledger = _feature_predictions(system, generated)
    report.ledger_rows = upsert_prediction_ledger(
        db_path, ledger, source="rebuilt_sequential_live"
    )

    combined = pd.concat([base_rows, normalized], ignore_index=True, sort=False)
    combined = combined.drop_duplicates("match_id", keep="last")
    sort_columns = [
        column
        for column in ("date", "event_date", "match_num", "match_id")
        if column in combined.columns
    ]
    combined.sort_values(sort_columns).reset_index(drop=True).to_parquet(
        processed_path, index=False
    )
    system.save(live_model_path)
    return report


def _resolve_identity(
    config: AppConfig,
    system: TennisPredictionSystem,
    provider_name: str,
    tour: str,
) -> PlayerIdentity | None:
    if not provider_name.strip():
        return None

    # New ID-first path: exact aliases and high-confidence fuzzy matches resolve
    # to immutable API-Tennis IDs. Unresolved names are queued for manual review.
    history_config = getattr(config, "history", None)
    if history_config is not None:
        identity_store = IdentityStore(history_config.identity_database_path)
        resolution = identity_store.resolve_name(
            provider_name,
            source="polymarket",
            tour=tour,
            fuzzy_threshold=getattr(history_config, "fuzzy_match_threshold", 96.0),
            minimum_margin=getattr(history_config, "fuzzy_minimum_margin", 3.0),
            queue_unresolved=True,
        )
        if resolution.resolved and system.name_resolver is not None:
            model_id = identity_store.model_player_id(resolution.provider_player_id)
            identity = system.name_resolver.by_id.get(model_id)
            if identity is not None:
                return identity

    # Backward-compatible path for the currently deployed pre-ID-first model.
    alias = resolve_player_alias(
        config.paper_trading.database_path,
        provider="polymarket",
        provider_name=provider_name,
        tour=tour,
    )
    if alias and system.name_resolver is not None:
        identity = system.name_resolver.by_id.get(alias["player_id"])
        if identity is not None:
            return identity
    resolver = system.name_resolver
    if resolver is None:
        return None
    attempts: list[str | None] = [tour] if tour else ["ATP", "WTA", None]
    for candidate_tour in attempts:
        try:
            return resolver.resolve(provider_name, tour=candidate_tour, threshold=88.0)
        except (KeyError, ValueError):
            continue
    return None


def _historical_surface(
    historical: pd.DataFrame,
    tournament: str,
    tour: str,
) -> tuple[str, str] | None:
    if historical.empty or "surface" not in historical or "tournament" not in historical:
        return None
    frame = historical[historical["surface"].isin(["Hard", "Clay", "Grass", "Carpet"])].copy()
    if tour and "tour" in frame:
        same_tour = frame[frame["tour"].astype(str).str.upper().eq(tour.upper())]
        if not same_tour.empty:
            frame = same_tour
    if frame.empty:
        return None
    frame["normalized_tournament"] = frame["tournament"].map(normalize_name)
    target = normalize_name(tournament)
    exact = frame[frame["normalized_tournament"].eq(target)]
    if exact.empty:
        choices = frame["normalized_tournament"].dropna().unique().tolist()
        found = process.extractOne(target, choices, scorer=fuzz.WRatio)
        if found is None or found[1] < 88:
            return None
        exact = frame[frame["normalized_tournament"].eq(found[0])]
    row = (
        exact.groupby(["surface", "tier"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
        .iloc[0]
    )
    return str(row["surface"]), str(row["tier"])


def _resolve_surface_tier(
    config: AppConfig,
    historical: pd.DataFrame,
    tournament: str,
    tour: str,
    inferred_tier: str,
) -> tuple[str | None, str]:
    saved = resolve_tournament_metadata(
        config.paper_trading.database_path,
        provider="polymarket",
        tournament=tournament,
        tour=tour,
    )
    if saved:
        return saved["surface"], saved["tier"]
    historical_value = _historical_surface(historical, tournament, tour)
    if historical_value:
        return historical_value
    return None, inferred_tier or "Other"


def _valid_price(value: object) -> float | None:
    price = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(price) or not 0 < float(price) < 1:
        return None
    return float(price)


def build_live_board(
    config: AppConfig,
    *,
    system: TennisPredictionSystem,
    board_date: str | pd.Timestamp | None = None,
    polymarket_client: PolymarketClient | None = None,
) -> LiveBoard:
    """Return every active Polymarket tennis market, even before mapping."""
    config.ensure_directories()
    db_path = config.paper_trading.database_path
    initialize_live_store(db_path)
    historical_path = Path(config.data.processed_path)
    historical = (
        pd.read_parquet(historical_path) if historical_path.exists() else pd.DataFrame()
    )
    client = polymarket_client or PolymarketClient()
    errors: list[str] = []
    selected_day = pd.Timestamp(
        board_date or pd.Timestamp.now(tz=config.live_data.timezone)
    ).date()
    local_start = pd.Timestamp(selected_day).tz_localize(config.live_data.timezone)
    local_end = local_start + pd.Timedelta(days=1)
    try:
        markets = client.fetch_active_tennis_markets(
            include_books=True,
            page_size=config.live_data.polymarket_page_size,
            max_pages=config.live_data.polymarket_market_pages,
            start_date=local_start.tz_convert("UTC"),
            end_date=local_end.tz_convert("UTC"),
            max_markets=getattr(config.live_data, "polymarket_max_markets_per_day", 60),
            book_batch_size=getattr(config.live_data, "polymarket_book_batch_size", 50),
        )
    except Exception as exc:
        markets = pd.DataFrame()
        errors.append(f"Polymarket refresh failed: {exc}")
    if markets.empty:
        return LiveBoard(
            markets=markets,
            matches=markets,
            quotes=markets,
            unresolved=pd.DataFrame(),
            errors=errors,
            diagnostics=getattr(client, "last_diagnostics", {}) or {},
        )

    # Gamma timestamps are UTC. Convert them to the configured local timezone and
    # keep only the exact calendar day selected by the user. Markets with missing
    # or invalid start times are excluded from a day-specific board rather than
    # being shown under the wrong date.
    markets = markets.copy()
    commence_utc = pd.to_datetime(markets.get("commence_time"), errors="coerce", utc=True)
    markets["commence_time"] = commence_utc.dt.tz_convert(config.live_data.timezone)
    markets = markets[markets["commence_time"].dt.date.eq(selected_day)].copy()
    diagnostics = getattr(client, "last_diagnostics", {}) or {}
    diagnostics = dict(diagnostics)
    for discovery_error in diagnostics.get("discovery_errors", []):
        errors.append(f"Polymarket discovery warning: {discovery_error}")
    diagnostics["selected_board_date"] = selected_day.isoformat()
    diagnostics["markets_on_selected_date"] = len(markets)
    if markets.empty:
        return LiveBoard(
            markets=markets,
            matches=markets,
            quotes=markets,
            unresolved=pd.DataFrame(),
            errors=errors,
            diagnostics=diagnostics,
        )

    upsert_polymarket_markets(db_path, markets)
    rows: list[dict[str, Any]] = []
    for market in markets.to_dict("records"):
        output = dict(market)
        tour = str(market.get("tour") or "")
        tier = str(market.get("tier") or "Other")
        tournament = str(market.get("tournament") or "Polymarket Tennis")
        player1 = str(market.get("player1") or "")
        player2 = str(market.get("player2") or "")

        identity1 = _resolve_identity(config, system, player1, tour)
        identity2 = _resolve_identity(config, system, player2, tour)
        surface, tier = _resolve_surface_tier(
            config, historical, tournament, tour, tier
        )

        status = "READY"
        reason = "Players mapped, surface resolved, and sufficient history recorded."
        if str(market.get("parse_status")) != "PARSED" or not player1 or not player2:
            status = "NEEDS MAPPING"
            reason = str(market.get("parse_reason") or "Map both player outcomes.")
        elif identity1 is None or identity2 is None:
            status = "NEEDS MAPPING"
            missing = [
                name
                for name, identity in ((player1, identity1), (player2, identity2))
                if identity is None
            ]
            reason = "Map to local player: " + ", ".join(missing)
        elif surface is None:
            status = "NEEDS SURFACE"
            reason = "Choose this tournament's court surface once."
        sample_quality = None
        if identity1 is not None and identity2 is not None and surface is not None:
            state1 = system.feature_engineer.state.players.get(identity1.player_id)
            state2 = system.feature_engineer.state.players.get(identity2.player_id)
            live_cfg = config.live_data
            sample_quality = evaluate_match_sample_quality(
                state1,
                state2,
                player1=identity1.name,
                player2=identity2.name,
                surface=str(surface),
                match_date=market.get("commence_time") or selected_day,
                absolute_min_matches=getattr(live_cfg, "cold_start_min_matches", 8),
                recommended_matches=getattr(live_cfg, "recommended_prediction_matches", 20),
                recommended_surface_matches=getattr(live_cfg, "recommended_surface_matches", 5),
                max_staleness_days=getattr(live_cfg, "max_player_staleness_days", 120),
            )
            if sample_quality.status == "INSUFFICIENT":
                status = "COLD START / NO HISTORY"
                reason = sample_quality.reason
            elif sample_quality.status == "LIMITED":
                status = "LIMITED SAMPLE / NO BET"
                reason = sample_quality.reason

        output.update(
            {
                "model_status": status,
                "model_reason": reason,
                "mapped_player1": identity1.name if identity1 else "",
                "mapped_player2": identity2.name if identity2 else "",
                "mapped_player1_id": identity1.player_id if identity1 else "",
                "mapped_player2_id": identity2.player_id if identity2 else "",
                "surface": surface or "Unknown",
                "tier": tier,
                "p_model1": np.nan,
                "p_elo1": np.nan,
                "model_as_of": getattr(system.feature_engineer.state, "as_of", None),
                "total_matches1": 0,
                "total_matches2": 0,
                "surface_matches1": 0,
                "surface_matches2": 0,
                "last_match_date1": None,
                "last_match_date2": None,
                "sample_status": "UNKNOWN",
                "sample_reason": "Player identities or surface are unresolved.",
                "sample_can_bet": False,
            }
        )
        if sample_quality is not None:
            output.update(
                {
                    "total_matches1": sample_quality.player1.total_matches,
                    "total_matches2": sample_quality.player2.total_matches,
                    "surface_matches1": sample_quality.player1.surface_matches,
                    "surface_matches2": sample_quality.player2.surface_matches,
                    "last_match_date1": sample_quality.player1.last_match_date,
                    "last_match_date2": sample_quality.player2.last_match_date,
                    "sample_status": sample_quality.status,
                    "sample_reason": sample_quality.reason,
                    "sample_can_bet": sample_quality.can_bet,
                }
            )

        if status in {"READY", "LIMITED SAMPLE / NO BET"}:
            match_time = pd.to_datetime(market.get("commence_time"), errors="coerce")
            if pd.isna(match_time):
                match_time = pd.Timestamp.today().normalize()
            try:
                prediction = system.predict_match(
                    player1=identity1.name,
                    player2=identity2.name,
                    surface=str(surface),
                    tournament_tier=tier,
                    match_date=match_time,
                    tour=tour or identity1.tour,
                )
                output["p_model1"] = float(prediction["player1_win_probability"])
                output["p_elo1"] = float(prediction["elo_probability"])
            except Exception as exc:
                output["model_status"] = "MODEL ERROR"
                output["model_reason"] = str(exc)

        ask1 = _valid_price(output.get("best_ask1"))
        ask2 = _valid_price(output.get("best_ask2"))
        p_model1 = _valid_price(output.get("p_model1"))
        output["decimal_odds1"] = 1.0 / ask1 if ask1 else np.nan
        output["decimal_odds2"] = 1.0 / ask2 if ask2 else np.nan
        output["edge1"] = p_model1 - ask1 if p_model1 is not None and ask1 else np.nan
        output["edge2"] = (1.0 - p_model1) - ask2 if p_model1 is not None and ask2 else np.nan
        if ask1 and ask2:
            total = ask1 + ask2
            output["fair_market_p1"] = ask1 / total
            output["fair_market_p2"] = ask2 / total
        else:
            output["fair_market_p1"] = np.nan
            output["fair_market_p2"] = np.nan
        rows.append(output)

    frame = pd.DataFrame(rows)
    if "commence_time" in frame:
        frame["commence_time"] = pd.to_datetime(frame["commence_time"], errors="coerce")
        frame = frame.sort_values(
            ["commence_time", "tournament", "display_title"], na_position="last"
        )
    frame = frame.reset_index(drop=True)
    return LiveBoard(
        markets=frame,
        matches=frame,
        quotes=markets,
        unresolved=pd.DataFrame(),
        errors=errors,
        diagnostics=diagnostics,
    )


def settle_recent_pending_bets(
    config: AppConfig,
    *,
    days_back: int = 2,
    model_path: str | Path = "outputs/backtest/trained_system.joblib",
    client: ApiTennisClient | None = None,
) -> dict[str, int]:
    """Settle recent pending bets without mutating model state."""
    initialize_paper_trading_db(config.paper_trading.database_path)
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=max(int(days_back), 0))
    api = client or ApiTennisClient(api_key_env=config.live_data.api_tennis_key_env)
    raw = api.fetch_fixtures(start, end, timezone_name=config.live_data.timezone)
    resolver = None
    path = Path(model_path)
    if path.exists():
        resolver = TennisPredictionSystem.load(path).name_resolver
    player_resolver = ProviderPlayerResolver(
        config.live_data.player_aliases_path, resolver
    )
    settlements = normalize_api_tennis_settlements(
        raw, player_resolver=player_resolver
    )
    return settle_pending_bets_from_results(
        config.paper_trading.database_path, settlements
    )
