from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd

from tennis_predictor.data.names import normalize_name


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def _connection(path: str | Path) -> Iterator[sqlite3.Connection]:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_live_store(path: str | Path = "data/paper_trading.db") -> Path:
    db_path = Path(path)
    with _connection(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS match_history (
                match_id TEXT PRIMARY KEY,
                fixture_id TEXT UNIQUE,
                provider_event_id TEXT,
                date TEXT NOT NULL,
                commence_time TEXT,
                tour TEXT,
                tier TEXT,
                tournament TEXT,
                tournament_key TEXT,
                round TEXT,
                surface TEXT,
                surface_known INTEGER NOT NULL DEFAULT 0,
                player1 TEXT NOT NULL,
                player2 TEXT NOT NULL,
                player1_id TEXT,
                player2_id TEXT,
                player1_won REAL,
                score TEXT,
                event_status TEXT,
                retirement INTEGER NOT NULL DEFAULT 0,
                walkover INTEGER NOT NULL DEFAULT 0,
                void INTEGER NOT NULL DEFAULT 0,
                source_file TEXT,
                saved_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_match_history_date
                ON match_history(date);
            CREATE INDEX IF NOT EXISTS idx_match_history_tournament
                ON match_history(date, tournament);

            CREATE TABLE IF NOT EXISTS prediction_ledger (
                match_id TEXT PRIMARY KEY,
                predicted_at TEXT NOT NULL,
                model_as_of TEXT,
                player1 TEXT NOT NULL,
                player2 TEXT NOT NULL,
                p_model1 REAL,
                p_elo1 REAL,
                actual REAL,
                prediction_status TEXT NOT NULL DEFAULT 'READY',
                source TEXT NOT NULL,
                FOREIGN KEY(match_id) REFERENCES match_history(match_id)
            );

            CREATE TABLE IF NOT EXISTS player_aliases_live (
                provider TEXT NOT NULL,
                provider_name TEXT NOT NULL,
                normalized_provider_name TEXT NOT NULL,
                tour TEXT NOT NULL DEFAULT '',
                local_player_id TEXT NOT NULL,
                local_player_name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(provider, normalized_provider_name, tour)
            );

            CREATE TABLE IF NOT EXISTS tournament_metadata_live (
                provider TEXT NOT NULL,
                provider_tournament TEXT NOT NULL,
                normalized_tournament TEXT NOT NULL,
                tour TEXT NOT NULL DEFAULT '',
                surface TEXT NOT NULL,
                tier TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(provider, normalized_tournament, tour)
            );

            CREATE TABLE IF NOT EXISTS polymarket_markets (
                market_id TEXT PRIMARY KEY,
                condition_id TEXT,
                event_id TEXT,
                question TEXT,
                slug TEXT,
                market_url TEXT,
                commence_time TEXT,
                tournament TEXT,
                tour TEXT,
                tier TEXT,
                player1 TEXT,
                player2 TEXT,
                token_id1 TEXT,
                token_id2 TEXT,
                active INTEGER NOT NULL DEFAULT 1,
                raw_json TEXT,
                fetched_at TEXT NOT NULL
            );
            """
        )
    return db_path


def _iso(value: object) -> str | None:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    return pd.Timestamp(timestamp).isoformat()


def _bool_int(value: object) -> int:
    if pd.isna(value):
        return 0
    return int(bool(value))


def upsert_match_history(path: str | Path, matches: pd.DataFrame) -> int:
    initialize_live_store(path)
    if matches is None or matches.empty:
        return 0
    rows = []
    for row in matches.to_dict("records"):
        match_id = str(row.get("match_id") or row.get("fixture_id") or "").strip()
        player1 = str(row.get("player1") or "").strip()
        player2 = str(row.get("player2") or "").strip()
        date_value = _iso(row.get("date"))
        if not match_id or not player1 or not player2 or not date_value:
            continue
        surface = str(row.get("surface") or "Unknown").title()
        surface_known = surface in {"Hard", "Clay", "Grass", "Carpet"}
        rows.append(
            (
                match_id,
                str(row.get("fixture_id") or match_id),
                str(row.get("provider_event_id") or ""),
                date_value,
                _iso(row.get("commence_time")),
                str(row.get("tour") or ""),
                str(row.get("tier") or ""),
                str(row.get("tournament") or "Unknown"),
                str(row.get("tournament_key") or ""),
                str(row.get("round") or ""),
                surface,
                int(surface_known),
                player1,
                player2,
                str(row.get("player1_id") or ""),
                str(row.get("player2_id") or ""),
                None if pd.isna(row.get("player1_won")) else float(row.get("player1_won")),
                str(row.get("score") or ""),
                str(row.get("event_status") or ""),
                _bool_int(row.get("retirement", False)),
                _bool_int(row.get("walkover", False)),
                _bool_int(row.get("void", False)),
                str(row.get("source_file") or ""),
                _utc_now(),
            )
        )
    if not rows:
        return 0
    with _connection(path) as connection:
        connection.executemany(
            """
            INSERT INTO match_history(
                match_id, fixture_id, provider_event_id, date, commence_time,
                tour, tier, tournament, tournament_key, round, surface,
                surface_known, player1, player2, player1_id, player2_id,
                player1_won, score, event_status, retirement, walkover, void,
                source_file, saved_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(match_id) DO UPDATE SET
                fixture_id=excluded.fixture_id,
                provider_event_id=excluded.provider_event_id,
                date=excluded.date,
                commence_time=excluded.commence_time,
                tour=excluded.tour,
                tier=excluded.tier,
                tournament=excluded.tournament,
                tournament_key=excluded.tournament_key,
                round=excluded.round,
                surface=excluded.surface,
                surface_known=excluded.surface_known,
                player1=excluded.player1,
                player2=excluded.player2,
                player1_id=excluded.player1_id,
                player2_id=excluded.player2_id,
                player1_won=excluded.player1_won,
                score=excluded.score,
                event_status=excluded.event_status,
                retirement=excluded.retirement,
                walkover=excluded.walkover,
                void=excluded.void,
                source_file=excluded.source_file,
                saved_at=excluded.saved_at
            """,
            rows,
        )
    return len(rows)


def upsert_prediction_ledger(path: str | Path, predictions: pd.DataFrame, *, source: str) -> int:
    initialize_live_store(path)
    if predictions is None or predictions.empty:
        return 0
    rows = []
    for row in predictions.to_dict("records"):
        match_id = str(row.get("match_id") or "").strip()
        if not match_id:
            continue
        rows.append(
            (
                match_id,
                _iso(row.get("predicted_at")) or _utc_now(),
                _iso(row.get("model_as_of")),
                str(row.get("player1") or ""),
                str(row.get("player2") or ""),
                None if pd.isna(row.get("p_model1")) else float(row.get("p_model1")),
                None if pd.isna(row.get("p_elo1")) else float(row.get("p_elo1")),
                None if pd.isna(row.get("actual")) else float(row.get("actual")),
                str(row.get("prediction_status") or "READY"),
                source,
            )
        )
    if not rows:
        return 0
    with _connection(path) as connection:
        connection.executemany(
            """
            INSERT INTO prediction_ledger(
                match_id, predicted_at, model_as_of, player1, player2,
                p_model1, p_elo1, actual, prediction_status, source
            ) VALUES (?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(match_id) DO UPDATE SET
                predicted_at=excluded.predicted_at,
                model_as_of=excluded.model_as_of,
                player1=excluded.player1,
                player2=excluded.player2,
                p_model1=excluded.p_model1,
                p_elo1=excluded.p_elo1,
                actual=excluded.actual,
                prediction_status=excluded.prediction_status,
                source=excluded.source
            """,
            rows,
        )
    return len(rows)


def seed_walkforward_ledger(
    path: str | Path,
    predictions: pd.DataFrame,
) -> int:
    if predictions is None or predictions.empty:
        return 0
    frame = predictions.copy()
    frame = frame[frame.get("model", pd.Series(index=frame.index, dtype=str)).isin(["boosted", "elo_baseline"])]
    if frame.empty:
        return 0
    pivot = frame.pivot_table(index="match_id", columns="model", values="probability", aggfunc="first")
    metadata = frame.sort_values(["match_id", "model"]).drop_duplicates("match_id", keep="first").set_index("match_id")
    ledger = metadata[[column for column in ["player1", "player2", "actual", "date"] if column in metadata.columns]].copy()
    ledger["p_model1"] = pivot.get("boosted")
    ledger["p_elo1"] = pivot.get("elo_baseline")
    ledger["predicted_at"] = ledger.get("date")
    ledger["model_as_of"] = ledger.get("date")
    ledger["prediction_status"] = "READY"
    ledger = ledger.reset_index()
    return upsert_prediction_ledger(path, ledger, source="walk_forward")


def available_history_dates(path: str | Path) -> list[pd.Timestamp]:
    initialize_live_store(path)
    with _connection(path) as connection:
        rows = connection.execute(
            "SELECT DISTINCT substr(date, 1, 10) AS d FROM match_history ORDER BY d"
        ).fetchall()
    return [pd.Timestamp(row["d"]) for row in rows if row["d"]]


def available_tournaments(path: str | Path, match_date: object) -> list[str]:
    initialize_live_store(path)
    day = pd.Timestamp(match_date).date().isoformat()
    with _connection(path) as connection:
        rows = connection.execute(
            """
            SELECT DISTINCT tournament FROM match_history
            WHERE substr(date, 1, 10) = ?
            ORDER BY tournament
            """,
            (day,),
        ).fetchall()
    return [str(row["tournament"]) for row in rows]


def history_matches(
    path: str | Path,
    *,
    match_date: object,
    tournament: str | None = None,
) -> pd.DataFrame:
    initialize_live_store(path)
    day = pd.Timestamp(match_date).date().isoformat()
    query = """
        SELECT h.*, p.predicted_at, p.model_as_of, p.p_model1, p.p_elo1,
               p.actual AS ledger_actual, p.prediction_status, p.source AS prediction_source
        FROM match_history h
        LEFT JOIN prediction_ledger p ON p.match_id = h.match_id
        WHERE substr(h.date, 1, 10) = ?
    """
    params: list[object] = [day]
    if tournament:
        query += " AND h.tournament = ?"
        params.append(tournament)
    query += " ORDER BY COALESCE(h.commence_time, h.date), h.player1, h.player2"
    with _connection(path) as connection:
        return pd.read_sql_query(query, connection, params=params)


def save_player_alias(
    path: str | Path,
    *,
    provider: str,
    provider_name: str,
    tour: str,
    local_player_id: str,
    local_player_name: str,
) -> None:
    initialize_live_store(path)
    with _connection(path) as connection:
        connection.execute(
            """
            INSERT INTO player_aliases_live(
                provider, provider_name, normalized_provider_name, tour,
                local_player_id, local_player_name, created_at
            ) VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(provider, normalized_provider_name, tour) DO UPDATE SET
                provider_name=excluded.provider_name,
                local_player_id=excluded.local_player_id,
                local_player_name=excluded.local_player_name,
                created_at=excluded.created_at
            """,
            (
                provider,
                provider_name,
                normalize_name(provider_name),
                str(tour or "").upper(),
                local_player_id,
                local_player_name,
                _utc_now(),
            ),
        )


def resolve_player_alias(
    path: str | Path,
    *,
    provider: str,
    provider_name: str,
    tour: str,
) -> dict[str, str] | None:
    initialize_live_store(path)
    with _connection(path) as connection:
        row = connection.execute(
            """
            SELECT local_player_id, local_player_name FROM player_aliases_live
            WHERE provider = ? AND normalized_provider_name = ? AND tour = ?
            """,
            (provider, normalize_name(provider_name), str(tour or "").upper()),
        ).fetchone()
        if row is None and tour:
            row = connection.execute(
                """
                SELECT local_player_id, local_player_name FROM player_aliases_live
                WHERE provider = ? AND normalized_provider_name = ? AND tour = ''
                """,
                (provider, normalize_name(provider_name)),
            ).fetchone()
    if row is None:
        return None
    return {"player_id": str(row["local_player_id"]), "name": str(row["local_player_name"])}


def save_tournament_metadata(
    path: str | Path,
    *,
    provider: str,
    tournament: str,
    tour: str,
    surface: str,
    tier: str,
) -> None:
    surface = str(surface).title()
    if surface not in {"Hard", "Clay", "Grass", "Carpet"}:
        raise ValueError("surface must be Hard, Clay, Grass, or Carpet")
    initialize_live_store(path)
    with _connection(path) as connection:
        connection.execute(
            """
            INSERT INTO tournament_metadata_live(
                provider, provider_tournament, normalized_tournament,
                tour, surface, tier, updated_at
            ) VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(provider, normalized_tournament, tour) DO UPDATE SET
                provider_tournament=excluded.provider_tournament,
                surface=excluded.surface,
                tier=excluded.tier,
                updated_at=excluded.updated_at
            """,
            (
                provider,
                tournament,
                normalize_name(tournament),
                str(tour or "").upper(),
                surface,
                tier,
                _utc_now(),
            ),
        )


def resolve_tournament_metadata(
    path: str | Path,
    *,
    provider: str,
    tournament: str,
    tour: str,
) -> dict[str, str] | None:
    initialize_live_store(path)
    normalized = normalize_name(tournament)
    with _connection(path) as connection:
        row = connection.execute(
            """
            SELECT surface, tier FROM tournament_metadata_live
            WHERE provider = ? AND normalized_tournament = ? AND tour = ?
            """,
            (provider, normalized, str(tour or "").upper()),
        ).fetchone()
        if row is None:
            row = connection.execute(
                """
                SELECT surface, tier FROM tournament_metadata_live
                WHERE provider = ? AND normalized_tournament = ? AND tour = ''
                """,
                (provider, normalized),
            ).fetchone()
    if row is None:
        return None
    return {"surface": str(row["surface"]), "tier": str(row["tier"])}


def upsert_polymarket_markets(path: str | Path, markets: pd.DataFrame) -> int:
    initialize_live_store(path)
    if markets is None or markets.empty:
        return 0
    rows = []
    for row in markets.to_dict("records"):
        market_id = str(row.get("market_id") or "").strip()
        if not market_id:
            continue
        raw = row.get("raw_json")
        if not isinstance(raw, str):
            raw = json.dumps(raw or {}, default=str)
        rows.append(
            (
                market_id,
                str(row.get("condition_id") or ""),
                str(row.get("provider_event_id") or row.get("event_id") or ""),
                str(row.get("question") or ""),
                str(row.get("market_slug") or row.get("slug") or ""),
                str(row.get("market_url") or ""),
                _iso(row.get("commence_time")),
                str(row.get("tournament") or "Polymarket Tennis"),
                str(row.get("tour") or ""),
                str(row.get("tier") or ""),
                str(row.get("player1") or ""),
                str(row.get("player2") or ""),
                str(row.get("token_id1") or ""),
                str(row.get("token_id2") or ""),
                int(bool(row.get("active", True))),
                raw,
                _utc_now(),
            )
        )
    if not rows:
        return 0
    with _connection(path) as connection:
        connection.executemany(
            """
            INSERT INTO polymarket_markets(
                market_id, condition_id, event_id, question, slug, market_url,
                commence_time, tournament, tour, tier, player1, player2,
                token_id1, token_id2, active, raw_json, fetched_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(market_id) DO UPDATE SET
                condition_id=excluded.condition_id,
                event_id=excluded.event_id,
                question=excluded.question,
                slug=excluded.slug,
                market_url=excluded.market_url,
                commence_time=excluded.commence_time,
                tournament=excluded.tournament,
                tour=excluded.tour,
                tier=excluded.tier,
                player1=excluded.player1,
                player2=excluded.player2,
                token_id1=excluded.token_id1,
                token_id2=excluded.token_id2,
                active=excluded.active,
                raw_json=excluded.raw_json,
                fetched_at=excluded.fetched_at
            """,
            rows,
        )
    return len(rows)


def history_range(
    path: str | Path,
    *,
    start_date: object,
    end_date: object,
) -> pd.DataFrame:
    initialize_live_store(path)
    start = pd.Timestamp(start_date).date().isoformat()
    end = pd.Timestamp(end_date).date().isoformat()
    with _connection(path) as connection:
        return pd.read_sql_query(
            """
            SELECT * FROM match_history
            WHERE substr(date,1,10) BETWEEN ? AND ?
            ORDER BY date, COALESCE(commence_time,date), match_id
            """,
            connection,
            params=(start, end),
        )
