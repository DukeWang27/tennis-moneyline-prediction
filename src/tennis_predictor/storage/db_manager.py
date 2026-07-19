from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DatabaseManager:
    """Small SQLite helper used by the canonical identity and match stores.

    Every write runs in an explicit transaction, foreign keys are enabled, and
    WAL mode keeps Streamlit reads from blocking ingestion writes.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=60)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 60000")
        try:
            connection.execute("BEGIN")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def table_exists(self, connection: sqlite3.Connection, name: str) -> bool:
        row = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (name,),
        ).fetchone()
        return row is not None

    def view_exists(self, connection: sqlite3.Connection, name: str) -> bool:
        row = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='view' AND name=?",
            (name,),
        ).fetchone()
        return row is not None

    def columns(self, connection: sqlite3.Connection, table: str) -> set[str]:
        if not self.table_exists(connection, table):
            return set()
        return {
            str(row["name"])
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }

    def initialize_core_schema(self) -> None:
        """Create the non-destructive canonical schema and ingestion audit tables."""
        with self.connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS players (
                    canonical_player_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    canonical_name TEXT NOT NULL,
                    normalized_name TEXT NOT NULL,
                    dob TEXT,
                    nationality TEXT,
                    tour TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_players_normalized_name
                    ON players(normalized_name, tour);

                CREATE TABLE IF NOT EXISTS player_source_mappings (
                    canonical_player_id INTEGER NOT NULL,
                    source_name TEXT NOT NULL,
                    source_player_id TEXT NOT NULL,
                    raw_name TEXT,
                    verified INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(source_name, source_player_id),
                    FOREIGN KEY(canonical_player_id)
                        REFERENCES players(canonical_player_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_source_mappings_player
                    ON player_source_mappings(canonical_player_id);

                CREATE TABLE IF NOT EXISTS player_aliases (
                    alias_name TEXT PRIMARY KEY,
                    display_alias TEXT NOT NULL,
                    canonical_player_id INTEGER NOT NULL,
                    source_name TEXT NOT NULL,
                    confidence REAL,
                    verified INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(canonical_player_id)
                        REFERENCES players(canonical_player_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_aliases_player
                    ON player_aliases(canonical_player_id);

                CREATE TABLE IF NOT EXISTS alias_review_queue (
                    review_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_name TEXT NOT NULL,
                    source_player_id TEXT,
                    raw_name TEXT NOT NULL,
                    normalized_name TEXT NOT NULL,
                    tour TEXT NOT NULL,
                    suggested_canonical_player_id INTEGER,
                    suggested_name TEXT,
                    fuzzy_score REAL,
                    second_score REAL,
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    reason TEXT,
                    created_at TEXT NOT NULL,
                    resolved_at TEXT,
                    resolved_canonical_player_id INTEGER,
                    UNIQUE(source_name, source_player_id, normalized_name, tour, status),
                    FOREIGN KEY(suggested_canonical_player_id)
                        REFERENCES players(canonical_player_id),
                    FOREIGN KEY(resolved_canonical_player_id)
                        REFERENCES players(canonical_player_id)
                );

                CREATE INDEX IF NOT EXISTS idx_alias_review_status
                    ON alias_review_queue(status, created_at);

                CREATE TABLE IF NOT EXISTS tournament_registry (
                    provider TEXT NOT NULL,
                    provider_tournament_id TEXT NOT NULL,
                    canonical_name TEXT NOT NULL,
                    normalized_name TEXT NOT NULL,
                    tour TEXT NOT NULL,
                    surface TEXT NOT NULL DEFAULT 'Unknown',
                    tier TEXT NOT NULL,
                    source_updated_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(provider, provider_tournament_id, tour)
                );

                CREATE INDEX IF NOT EXISTS idx_tournament_registry_name
                    ON tournament_registry(provider, normalized_name, tour);

                CREATE TABLE IF NOT EXISTS source_ingestion_state (
                    source_name TEXT PRIMARY KEY,
                    dataset_slug TEXT,
                    remote_manifest_hash TEXT,
                    remote_version_marker TEXT,
                    archive_path TEXT,
                    schema_hash TEXT,
                    max_match_date TEXT,
                    row_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    details_json TEXT,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS canonical_matches (
                    match_fingerprint TEXT PRIMARY KEY,
                    date_time TEXT NOT NULL,
                    match_date TEXT NOT NULL,
                    canonical_player1_id INTEGER NOT NULL,
                    canonical_player2_id INTEGER NOT NULL,
                    tournament_name TEXT NOT NULL,
                    round_name TEXT NOT NULL,
                    tour TEXT,
                    tier TEXT,
                    surface TEXT,
                    player1_won INTEGER,
                    score TEXT,
                    status TEXT,
                    retirement INTEGER NOT NULL DEFAULT 0,
                    walkover INTEGER NOT NULL DEFAULT 0,
                    defaulted INTEGER NOT NULL DEFAULT 0,
                    void INTEGER NOT NULL DEFAULT 0,
                    sources_present TEXT NOT NULL,
                    field_provenance TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(canonical_player1_id)
                        REFERENCES players(canonical_player_id),
                    FOREIGN KEY(canonical_player2_id)
                        REFERENCES players(canonical_player_id)
                );

                CREATE TABLE IF NOT EXISTS match_source_records (
                    source_name TEXT NOT NULL,
                    source_match_id TEXT NOT NULL,
                    match_fingerprint TEXT NOT NULL,
                    raw_record_json TEXT NOT NULL,
                    imported_at TEXT NOT NULL,
                    PRIMARY KEY(source_name, source_match_id),
                    FOREIGN KEY(match_fingerprint)
                        REFERENCES canonical_matches(match_fingerprint) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_match_source_fingerprint
                    ON match_source_records(match_fingerprint);

                CREATE TABLE IF NOT EXISTS closing_odds (
                    match_fingerprint TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    bookmaker TEXT NOT NULL DEFAULT 'consensus',
                    player1_decimal REAL,
                    player2_decimal REAL,
                    observed_at TEXT,
                    is_closing INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(match_fingerprint, source_name, bookmaker),
                    FOREIGN KEY(match_fingerprint)
                        REFERENCES canonical_matches(match_fingerprint) ON DELETE CASCADE
                );
                """
            )
