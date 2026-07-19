from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz, process

from tennis_predictor.data.names import normalize_name
from .db_manager import DatabaseManager, utc_now


_SOURCE_ALIASES = {
    "api-tennis": "api_tennis",
    "api_tennis": "api_tennis",
    "apitennis": "api_tennis",
    "huge-tennis": "huge_tennis",
    "huge_tennis": "huge_tennis",
    "kaggle-atp-daily": "kaggle_atp_daily",
    "kaggle_atp_daily": "kaggle_atp_daily",
    "kaggle-wta-daily": "kaggle_wta_daily",
    "kaggle_wta_daily": "kaggle_wta_daily",
    "tennis-data": "tennis_data",
    "tennis_data": "tennis_data",
    "sackmann": "sackmann",
    "polymarket": "polymarket",
}


def normalize_source_name(value: str) -> str:
    raw = str(value or "").strip().casefold().replace(" ", "_")
    return _SOURCE_ALIASES.get(raw, raw.replace("-", "_"))


def _legacy_table_name(connection: sqlite3.Connection, base: str) -> str:
    candidate = f"legacy_{base}_api_id"
    suffix = 1
    while connection.execute(
        "SELECT 1 FROM sqlite_master WHERE name=?", (candidate,)
    ).fetchone():
        candidate = f"legacy_{base}_api_id_{suffix}"
        suffix += 1
    return candidate


def _migrate_legacy_api_id_schema(manager: DatabaseManager) -> None:
    """Migrate the previous API-Tennis-ID-as-PK schema without deleting rows.

    The old tables are renamed and retained. Every old provider ID receives a
    new canonical player row plus an ``api_tennis`` source mapping.
    """

    with manager.connection() as connection:
        player_columns = manager.columns(connection, "players")
        if "provider_player_id" not in player_columns or "canonical_player_id" in player_columns:
            return

        renamed: dict[str, str] = {}
        for table in ("player_aliases", "source_player_ids", "alias_review_queue", "players"):
            if manager.table_exists(connection, table):
                legacy = _legacy_table_name(connection, table)
                connection.execute(f'ALTER TABLE "{table}" RENAME TO "{legacy}"')
                renamed[table] = legacy

    manager.initialize_core_schema()

    with manager.connection() as connection:
        legacy_players = renamed.get("players")
        if not legacy_players:
            return
        rows = connection.execute(f'SELECT * FROM "{legacy_players}"').fetchall()
        provider_to_canonical: dict[int, int] = {}
        for row in rows:
            now = utc_now()
            connection.execute(
                """
                INSERT INTO players(
                    canonical_name, normalized_name, dob, nationality, tour,
                    created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?)
                """,
                (
                    str(row["canonical_name"]),
                    str(row["normalized_name"]),
                    row["dob"],
                    row["nationality"],
                    str(row["tour"] or ""),
                    row["created_at"] or now,
                    row["updated_at"] or now,
                ),
            )
            canonical_id = int(connection.execute("SELECT last_insert_rowid()").fetchone()[0])
            provider_id = int(row["provider_player_id"])
            provider_to_canonical[provider_id] = canonical_id
            connection.execute(
                """
                INSERT INTO player_source_mappings(
                    canonical_player_id, source_name, source_player_id, raw_name,
                    verified, created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?)
                """,
                (
                    canonical_id,
                    "api_tennis",
                    str(provider_id),
                    str(row["canonical_name"]),
                    1,
                    now,
                    now,
                ),
            )
            alias = normalize_name(str(row["canonical_name"]))
            if alias:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO player_aliases(
                        alias_name, display_alias, canonical_player_id, source_name,
                        confidence, verified, created_at, updated_at
                    ) VALUES (?,?,?,?,?,?,?,?)
                    """,
                    (
                        alias,
                        str(row["canonical_name"]),
                        canonical_id,
                        "api_tennis",
                        100.0,
                        1,
                        now,
                        now,
                    ),
                )

        legacy_aliases = renamed.get("player_aliases")
        if legacy_aliases:
            for row in connection.execute(f'SELECT * FROM "{legacy_aliases}"').fetchall():
                canonical_id = provider_to_canonical.get(int(row["provider_player_id"]))
                if canonical_id is None:
                    continue
                connection.execute(
                    """
                    INSERT OR IGNORE INTO player_aliases(
                        alias_name, display_alias, canonical_player_id, source_name,
                        confidence, verified, created_at, updated_at
                    ) VALUES (?,?,?,?,?,?,?,?)
                    """,
                    (
                        str(row["alias_name"]),
                        str(row["display_alias"]),
                        canonical_id,
                        normalize_source_name(str(row["source"])),
                        row["confidence"],
                        int(row["verified"] or 0),
                        row["created_at"] or utc_now(),
                        utc_now(),
                    ),
                )

        legacy_sources = renamed.get("source_player_ids")
        if legacy_sources:
            for row in connection.execute(f'SELECT * FROM "{legacy_sources}"').fetchall():
                canonical_id = provider_to_canonical.get(int(row["provider_player_id"]))
                if canonical_id is None:
                    continue
                now = utc_now()
                connection.execute(
                    """
                    INSERT INTO player_source_mappings(
                        canonical_player_id, source_name, source_player_id, raw_name,
                        verified, created_at, updated_at
                    ) VALUES (?,?,?,?,?,?,?)
                    ON CONFLICT(source_name, source_player_id) DO UPDATE SET
                        canonical_player_id=excluded.canonical_player_id,
                        updated_at=excluded.updated_at
                    """,
                    (
                        canonical_id,
                        normalize_source_name(str(row["source"])),
                        str(row["source_player_id"]),
                        None,
                        1,
                        row["created_at"] or now,
                        now,
                    ),
                )

        legacy_reviews = renamed.get("alias_review_queue")
        if legacy_reviews:
            for row in connection.execute(f'SELECT * FROM "{legacy_reviews}"').fetchall():
                suggested = row["suggested_player_id"]
                suggested_canonical = (
                    provider_to_canonical.get(int(suggested)) if suggested is not None else None
                )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO alias_review_queue(
                        source_name, source_player_id, raw_name, normalized_name, tour,
                        suggested_canonical_player_id, suggested_name, fuzzy_score,
                        status, reason, created_at, resolved_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        normalize_source_name(str(row["source"])),
                        None,
                        str(row["raw_name"]),
                        str(row["normalized_name"]),
                        str(row["tour"] or ""),
                        suggested_canonical,
                        row["suggested_name"],
                        row["fuzzy_score"],
                        str(row["status"] or "PENDING"),
                        "migrated from API-ID-first schema",
                        row["created_at"] or utc_now(),
                        row["resolved_at"],
                    ),
                )


def initialize_identity_store(path: str | Path) -> Path:
    manager = DatabaseManager(path)
    _migrate_legacy_api_id_schema(manager)
    manager.initialize_core_schema()
    return manager.path


@dataclass(frozen=True, slots=True)
class IdentityResolution:
    canonical_player_id: int | None
    canonical_name: str | None
    status: str
    score: float | None = None
    review_id: int | None = None
    provider_player_id: int | None = None
    source_name: str | None = None
    source_player_id: str | None = None

    @property
    def resolved(self) -> bool:
        return self.canonical_player_id is not None and self.status in {
            "EXACT_ID",
            "EXACT_ALIAS",
            "EXACT_NAME",
            "FUZZY_AUTO",
            "CREATED",
        }


class IdentityStore:
    """Canonical player registry with source-specific IDs and review queue."""

    def __init__(self, path: str | Path):
        self.path = initialize_identity_store(path)
        self.db = DatabaseManager(self.path)

    @staticmethod
    def model_player_id(provider_player_id: int | str) -> str:
        """Compatibility ID for the currently deployed API-Tennis-based model."""
        return f"api_tennis:{int(provider_player_id)}"

    @staticmethod
    def canonical_model_player_id(canonical_player_id: int | str) -> str:
        return f"canonical:{int(canonical_player_id)}"

    @staticmethod
    def compatible_tours(tour: str) -> set[str]:
        value = str(tour or "").upper()
        if value == "ATP":
            return {"ATP", "ITF_M"}
        if value == "ITF_M":
            return {"ITF_M", "ATP"}
        if value == "WTA":
            return {"WTA", "ITF_W"}
        if value == "ITF_W":
            return {"ITF_W", "WTA"}
        return {value} if value else set()

    def _api_provider_id(
        self, connection: sqlite3.Connection, canonical_player_id: int
    ) -> int | None:
        row = connection.execute(
            """
            SELECT source_player_id
            FROM player_source_mappings
            WHERE canonical_player_id=? AND source_name='api_tennis'
            """,
            (int(canonical_player_id),),
        ).fetchone()
        if row is None or not str(row["source_player_id"]).isdigit():
            return None
        return int(row["source_player_id"])

    def _resolution_from_row(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        status: str,
        score: float | None = None,
        *,
        source_name: str | None = None,
        source_player_id: str | None = None,
    ) -> IdentityResolution:
        canonical_id = int(row["canonical_player_id"])
        return IdentityResolution(
            canonical_player_id=canonical_id,
            canonical_name=str(row["canonical_name"]),
            status=status,
            score=score,
            provider_player_id=self._api_provider_id(connection, canonical_id),
            source_name=source_name,
            source_player_id=source_player_id,
        )

    def create_player(
        self,
        *,
        canonical_name: str,
        tour: str,
        dob: str | None = None,
        nationality: str | None = None,
    ) -> int:
        name = str(canonical_name).strip()
        if not name:
            raise ValueError("canonical_name cannot be blank")
        now = utc_now()
        with self.db.connection() as connection:
            connection.execute(
                """
                INSERT INTO players(
                    canonical_name, normalized_name, dob, nationality, tour,
                    created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?)
                """,
                (
                    name,
                    normalize_name(name),
                    dob,
                    nationality,
                    str(tour or "").upper(),
                    now,
                    now,
                ),
            )
            canonical_id = int(connection.execute("SELECT last_insert_rowid()").fetchone()[0])
        self.add_alias(
            name,
            canonical_id,
            source="canonical",
            confidence=100.0,
            verified=True,
            queue_conflict=False,
        )
        return canonical_id

    def upsert_player(
        self,
        provider_player_id: int | str,
        *,
        canonical_name: str,
        tour: str,
        dob: str | None = None,
        nationality: str | None = None,
        source_updated_at: str | None = None,
        aliases: list[str] | None = None,
    ) -> int:
        """Backward-compatible API-Tennis upsert.

        Returns the API-Tennis provider ID so older live code remains stable.
        The canonical ID is available through ``resolve_source_id``.
        """
        provider_id = str(int(provider_player_id))
        source_resolution = self.resolve_source_id(
            source="api_tennis", source_player_id=provider_id
        )
        if source_resolution.resolved:
            canonical_id = int(source_resolution.canonical_player_id)
            now = utc_now()
            with self.db.connection() as connection:
                connection.execute(
                    """
                    UPDATE players SET
                        canonical_name=CASE
                            WHEN length(?) > length(canonical_name) THEN ? ELSE canonical_name END,
                        normalized_name=CASE
                            WHEN length(?) > length(canonical_name) THEN ? ELSE normalized_name END,
                        dob=COALESCE(?, dob),
                        nationality=COALESCE(?, nationality),
                        tour=CASE WHEN tour='' THEN ? ELSE tour END,
                        updated_at=?
                    WHERE canonical_player_id=?
                    """,
                    (
                        canonical_name,
                        canonical_name,
                        canonical_name,
                        normalize_name(canonical_name),
                        dob,
                        nationality,
                        str(tour or "").upper(),
                        now,
                        canonical_id,
                    ),
                )
        else:
            exact = self.resolve_name(
                canonical_name,
                source="api_tennis",
                tour=tour,
                fuzzy_threshold=99.0,
                minimum_margin=5.0,
                queue_unresolved=False,
            )
            canonical_id = (
                int(exact.canonical_player_id)
                if exact.resolved
                else self.create_player(
                    canonical_name=canonical_name,
                    tour=tour,
                    dob=dob,
                    nationality=nationality,
                )
            )

        self.link_source_id(
            source="api_tennis",
            source_player_id=provider_id,
            canonical_player_id=canonical_id,
            raw_name=canonical_name,
            verified=True,
        )
        self.add_alias(
            canonical_name,
            canonical_id,
            source="api_tennis",
            confidence=100.0,
            verified=True,
            queue_conflict=False,
        )
        for alias in aliases or []:
            self.add_alias(
                alias,
                canonical_id,
                source="api_tennis",
                confidence=100.0,
                verified=True,
                queue_conflict=False,
            )
        return int(provider_player_id)

    def add_alias(
        self,
        alias_name: str,
        canonical_player_id: int | str,
        *,
        source: str,
        confidence: float | None = None,
        verified: bool = False,
        queue_conflict: bool = True,
    ) -> bool:
        alias_key = normalize_name(alias_name)
        if not alias_key:
            return False
        canonical_id = int(canonical_player_id)
        now = utc_now()
        with self.db.connection() as connection:
            existing = connection.execute(
                "SELECT canonical_player_id FROM player_aliases WHERE alias_name=?",
                (alias_key,),
            ).fetchone()
            if existing is not None and int(existing["canonical_player_id"]) != canonical_id:
                if queue_conflict:
                    player = connection.execute(
                        "SELECT canonical_name, tour FROM players WHERE canonical_player_id=?",
                        (canonical_id,),
                    ).fetchone()
                    self._queue_review_connection(
                        connection,
                        source=source,
                        source_player_id=None,
                        raw_name=alias_name,
                        tour=str(player["tour"] if player else ""),
                        suggested_canonical_player_id=canonical_id,
                        suggested_name=str(player["canonical_name"] if player else ""),
                        score=confidence,
                        second_score=None,
                        reason="alias collision",
                    )
                return False
            connection.execute(
                """
                INSERT INTO player_aliases(
                    alias_name, display_alias, canonical_player_id, source_name,
                    confidence, verified, created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?)
                ON CONFLICT(alias_name) DO UPDATE SET
                    display_alias=excluded.display_alias,
                    source_name=excluded.source_name,
                    confidence=COALESCE(excluded.confidence, player_aliases.confidence),
                    verified=MAX(player_aliases.verified, excluded.verified),
                    updated_at=excluded.updated_at
                """,
                (
                    alias_key,
                    str(alias_name).strip(),
                    canonical_id,
                    normalize_source_name(source),
                    confidence,
                    int(bool(verified)),
                    now,
                    now,
                ),
            )
        return True

    def link_source_id(
        self,
        *,
        source: str,
        source_player_id: str | int,
        canonical_player_id: str | int | None = None,
        provider_player_id: str | int | None = None,
        raw_name: str | None = None,
        verified: bool = True,
    ) -> None:
        source_name = normalize_source_name(source)
        canonical_id = int(canonical_player_id) if canonical_player_id is not None else None
        if canonical_id is None and provider_player_id is not None:
            if source_name == "api_tennis":
                existing = self.resolve_source_id(
                    source="api_tennis", source_player_id=provider_player_id
                )
                canonical_id = existing.canonical_player_id
            else:
                with self.db.connection() as connection:
                    direct = connection.execute(
                        "SELECT canonical_player_id FROM players WHERE canonical_player_id=?",
                        (int(provider_player_id),),
                    ).fetchone()
                    if direct is not None:
                        canonical_id = int(direct["canonical_player_id"])
                    else:
                        api = connection.execute(
                            """
                            SELECT canonical_player_id FROM player_source_mappings
                            WHERE source_name='api_tennis' AND source_player_id=?
                            """,
                            (str(provider_player_id),),
                        ).fetchone()
                        canonical_id = int(api["canonical_player_id"]) if api else None
        if canonical_id is None:
            raise ValueError("canonical_player_id could not be resolved")

        now = utc_now()
        with self.db.connection() as connection:
            connection.execute(
                """
                INSERT INTO player_source_mappings(
                    canonical_player_id, source_name, source_player_id, raw_name,
                    verified, created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?)
                ON CONFLICT(source_name, source_player_id) DO UPDATE SET
                    canonical_player_id=excluded.canonical_player_id,
                    raw_name=COALESCE(excluded.raw_name, player_source_mappings.raw_name),
                    verified=MAX(player_source_mappings.verified, excluded.verified),
                    updated_at=excluded.updated_at
                """,
                (
                    canonical_id,
                    source_name,
                    str(source_player_id),
                    raw_name,
                    int(bool(verified)),
                    now,
                    now,
                ),
            )

    def resolve_source_id(
        self, *, source: str, source_player_id: str | int
    ) -> IdentityResolution:
        source_name = normalize_source_name(source)
        with self.db.connection() as connection:
            row = connection.execute(
                """
                SELECT p.canonical_player_id, p.canonical_name
                FROM player_source_mappings m
                JOIN players p ON p.canonical_player_id=m.canonical_player_id
                WHERE m.source_name=? AND m.source_player_id=?
                """,
                (source_name, str(source_player_id)),
            ).fetchone()
            if row is None:
                return IdentityResolution(None, None, "UNRESOLVED")
            return self._resolution_from_row(
                connection,
                row,
                "EXACT_ID",
                100.0,
                source_name=source_name,
                source_player_id=str(source_player_id),
            )

    def resolve_name(
        self,
        raw_name: str,
        *,
        source: str,
        tour: str,
        fuzzy_threshold: float = 96.0,
        minimum_margin: float = 3.0,
        queue_unresolved: bool = True,
        source_player_id: str | int | None = None,
    ) -> IdentityResolution:
        normalized = normalize_name(raw_name)
        if not normalized:
            return IdentityResolution(None, None, "UNRESOLVED")
        source_name = normalize_source_name(source)
        tour = str(tour or "").upper()
        compatible_tours = self.compatible_tours(tour)
        with self.db.connection() as connection:
            alias_rows = connection.execute(
                """
                SELECT p.canonical_player_id, p.canonical_name, p.tour
                FROM player_aliases a
                JOIN players p ON p.canonical_player_id=a.canonical_player_id
                WHERE a.alias_name=?
                """,
                (normalized,),
            ).fetchall()
            if compatible_tours:
                alias_rows = [r for r in alias_rows if str(r["tour"]) in compatible_tours]
            if len(alias_rows) == 1:
                return self._resolution_from_row(
                    connection, alias_rows[0], "EXACT_ALIAS", 100.0,
                    source_name=source_name,
                    source_player_id=(str(source_player_id) if source_player_id is not None else None),
                )

            exact_rows = connection.execute(
                """
                SELECT canonical_player_id, canonical_name, tour
                FROM players WHERE normalized_name=?
                """,
                (normalized,),
            ).fetchall()
            if compatible_tours:
                exact_rows = [r for r in exact_rows if str(r["tour"]) in compatible_tours]
            if len(exact_rows) == 1:
                return self._resolution_from_row(
                    connection, exact_rows[0], "EXACT_NAME", 100.0,
                    source_name=source_name,
                    source_player_id=(str(source_player_id) if source_player_id is not None else None),
                )

            rows = connection.execute(
                "SELECT canonical_player_id, canonical_name, normalized_name, tour FROM players"
            ).fetchall()
            if compatible_tours:
                rows = [r for r in rows if str(r["tour"]) in compatible_tours]
            choices = {
                f"{int(r['canonical_player_id'])}:{str(r['normalized_name'])}": r
                for r in rows
            }
            ranked = process.extract(
                normalized,
                list(choices),
                scorer=lambda query, choice, **_: fuzz.WRatio(query, choice.split(":", 1)[1]),
                limit=2,
            )
            best_row = None
            best_score = None
            second_score = 0.0
            if ranked:
                best_key, best_score, _ = ranked[0]
                best_row = choices[best_key]
                second_score = float(ranked[1][1]) if len(ranked) > 1 else 0.0
            if (
                best_row is not None
                and float(best_score) >= float(fuzzy_threshold)
                and float(best_score) - second_score >= float(minimum_margin)
            ):
                canonical_id = int(best_row["canonical_player_id"])
                connection.execute(
                    """
                    INSERT OR IGNORE INTO player_aliases(
                        alias_name, display_alias, canonical_player_id, source_name,
                        confidence, verified, created_at, updated_at
                    ) VALUES (?,?,?,?,?,?,?,?)
                    """,
                    (
                        normalized,
                        str(raw_name).strip(),
                        canonical_id,
                        source_name,
                        float(best_score),
                        0,
                        utc_now(),
                        utc_now(),
                    ),
                )
                return self._resolution_from_row(
                    connection,
                    best_row,
                    "FUZZY_AUTO",
                    float(best_score),
                    source_name=source_name,
                    source_player_id=(str(source_player_id) if source_player_id is not None else None),
                )

            review_id = None
            if queue_unresolved:
                review_id = self._queue_review_connection(
                    connection,
                    source=source_name,
                    source_player_id=(str(source_player_id) if source_player_id is not None else None),
                    raw_name=raw_name,
                    tour=tour,
                    suggested_canonical_player_id=(
                        int(best_row["canonical_player_id"]) if best_row is not None else None
                    ),
                    suggested_name=(
                        str(best_row["canonical_name"]) if best_row is not None else None
                    ),
                    score=(float(best_score) if best_score is not None else None),
                    second_score=second_score,
                    reason="ambiguous or below fuzzy threshold",
                )
        return IdentityResolution(
            None,
            None,
            "REVIEW_REQUIRED" if queue_unresolved else "UNRESOLVED",
            float(best_score) if best_score is not None else None,
            review_id,
            None,
            source_name,
            str(source_player_id) if source_player_id is not None else None,
        )

    def resolve_or_create_source_player(
        self,
        *,
        source: str,
        source_player_id: str | int | None,
        raw_name: str,
        tour: str,
        dob: str | None = None,
        nationality: str | None = None,
        fuzzy_threshold: float = 97.0,
        minimum_margin: float = 4.0,
        review_floor: float = 82.0,
    ) -> IdentityResolution:
        source_name = normalize_source_name(source)
        if source_player_id is not None and str(source_player_id).strip():
            existing = self.resolve_source_id(
                source=source_name, source_player_id=str(source_player_id)
            )
            if existing.resolved:
                return existing

        # CLEANED_KAGGLE_SOURCE_IDS_ARE_AUTHORITATIVE
        #
        # The unified cleaning pipeline already provides a stable,
        # non-empty player source ID. Do not fuzzy-match these IDs
        # against unrelated or similarly named players.
        if (
            source_name.replace("-", "_")
            == "cleaned_kaggle_unified"
            and source_player_id is not None
            and str(source_player_id).strip()
        ):
            canonical_id = self.create_player(
                canonical_name=raw_name,
                tour=tour,
                dob=dob,
                nationality=nationality,
            )

            self.link_source_id(
                source=source_name,
                source_player_id=str(source_player_id),
                canonical_player_id=canonical_id,
                raw_name=raw_name,
                verified=True,
            )

            return self.resolve_source_id(
                source=source_name,
                source_player_id=str(source_player_id),
            )

        resolution = self.resolve_name(
            raw_name,
            source=source_name,
            tour=tour,
            fuzzy_threshold=fuzzy_threshold,
            minimum_margin=minimum_margin,
            queue_unresolved=False,
            source_player_id=source_player_id,
        )
        if resolution.resolved:
            canonical_id = int(resolution.canonical_player_id)
            if source_player_id is not None and str(source_player_id).strip():
                self.link_source_id(
                    source=source_name,
                    source_player_id=str(source_player_id),
                    canonical_player_id=canonical_id,
                    raw_name=raw_name,
                    verified=resolution.status in {"EXACT_ID", "EXACT_ALIAS", "EXACT_NAME"},
                )
            self.add_alias(
                raw_name,
                canonical_id,
                source=source_name,
                confidence=resolution.score,
                verified=resolution.status in {"EXACT_ALIAS", "EXACT_NAME"},
                queue_conflict=True,
            )
            return self.resolve_source_id(
                source=source_name,
                source_player_id=str(source_player_id),
            ) if source_player_id is not None and str(source_player_id).strip() else resolution

        # Avoid silently creating a likely duplicate when a moderately similar
        # candidate exists. Queue it for review instead.
        if resolution.score is not None and float(resolution.score) >= float(review_floor):
            queued = self.resolve_name(
                raw_name,
                source=source_name,
                tour=tour,
                fuzzy_threshold=fuzzy_threshold,
                minimum_margin=minimum_margin,
                queue_unresolved=True,
                source_player_id=source_player_id,
            )
            return queued

        canonical_id = self.create_player(
            canonical_name=raw_name,
            tour=tour,
            dob=dob,
            nationality=nationality,
        )
        if source_player_id is not None and str(source_player_id).strip():
            self.link_source_id(
                source=source_name,
                source_player_id=str(source_player_id),
                canonical_player_id=canonical_id,
                raw_name=raw_name,
                verified=False,
            )
        self.add_alias(
            raw_name,
            canonical_id,
            source=source_name,
            confidence=100.0,
            verified=False,
            queue_conflict=False,
        )
        with self.db.connection() as connection:
            row = connection.execute(
                "SELECT canonical_player_id, canonical_name FROM players WHERE canonical_player_id=?",
                (canonical_id,),
            ).fetchone()
            return self._resolution_from_row(
                connection,
                row,
                "CREATED",
                100.0,
                source_name=source_name,
                source_player_id=(str(source_player_id) if source_player_id is not None else None),
            )

    def upsert_tournament(
        self,
        *,
        provider: str,
        provider_tournament_id: str | int,
        canonical_name: str,
        tour: str,
        surface: str,
        tier: str,
        source_updated_at: str | None = None,
    ) -> None:
        normalized_surface = str(surface or "Unknown").title()
        if normalized_surface not in {"Hard", "Clay", "Grass", "Carpet", "Unknown"}:
            normalized_surface = "Unknown"
        now = utc_now()
        with self.db.connection() as connection:
            connection.execute(
                """
                INSERT INTO tournament_registry(
                    provider, provider_tournament_id, canonical_name,
                    normalized_name, tour, surface, tier, source_updated_at,
                    created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(provider, provider_tournament_id, tour) DO UPDATE SET
                    canonical_name=excluded.canonical_name,
                    normalized_name=excluded.normalized_name,
                    surface=CASE WHEN excluded.surface='Unknown'
                        THEN tournament_registry.surface ELSE excluded.surface END,
                    tier=excluded.tier,
                    source_updated_at=COALESCE(
                        excluded.source_updated_at, tournament_registry.source_updated_at
                    ),
                    updated_at=excluded.updated_at
                """,
                (
                    str(provider),
                    str(provider_tournament_id),
                    str(canonical_name),
                    normalize_name(canonical_name),
                    str(tour or "").upper(),
                    normalized_surface,
                    str(tier),
                    source_updated_at,
                    now,
                    now,
                ),
            )

    def resolve_tournament(
        self,
        *,
        provider: str,
        provider_tournament_id: str | int,
        tournament_name: str,
        tour: str,
    ) -> dict[str, str] | None:
        with self.db.connection() as connection:
            row = connection.execute(
                """
                SELECT surface, tier, canonical_name
                FROM tournament_registry
                WHERE provider=? AND provider_tournament_id=? AND tour=?
                """,
                (str(provider), str(provider_tournament_id), str(tour or "").upper()),
            ).fetchone()
            if row is None:
                row = connection.execute(
                    """
                    SELECT surface, tier, canonical_name
                    FROM tournament_registry
                    WHERE provider=? AND normalized_name=? AND tour=?
                    ORDER BY updated_at DESC LIMIT 1
                    """,
                    (
                        str(provider),
                        normalize_name(tournament_name),
                        str(tour or "").upper(),
                    ),
                ).fetchone()
        return dict(row) if row is not None else None

    def _queue_review_connection(
        self,
        connection: sqlite3.Connection,
        *,
        source: str,
        source_player_id: str | None,
        raw_name: str,
        tour: str,
        suggested_canonical_player_id: int | None,
        suggested_name: str | None,
        score: float | None,
        second_score: float | None,
        reason: str,
    ) -> int:
        normalized = normalize_name(raw_name)
        source_name = normalize_source_name(source)
        source_id = str(source_player_id or "")
        connection.execute(
            """
            INSERT INTO alias_review_queue(
                source_name, source_player_id, raw_name, normalized_name, tour,
                suggested_canonical_player_id, suggested_name, fuzzy_score,
                second_score, status, reason, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(source_name, source_player_id, normalized_name, tour, status)
            DO UPDATE SET
                raw_name=excluded.raw_name,
                suggested_canonical_player_id=excluded.suggested_canonical_player_id,
                suggested_name=excluded.suggested_name,
                fuzzy_score=excluded.fuzzy_score,
                second_score=excluded.second_score,
                reason=excluded.reason
            """,
            (
                source_name,
                source_id,
                str(raw_name).strip(),
                normalized,
                str(tour or "").upper(),
                suggested_canonical_player_id,
                suggested_name,
                score,
                second_score,
                "PENDING",
                reason,
                utc_now(),
            ),
        )
        row = connection.execute(
            """
            SELECT review_id FROM alias_review_queue
            WHERE source_name=? AND source_player_id=? AND normalized_name=?
              AND tour=? AND status='PENDING'
            """,
            (source_name, source_id, normalized, str(tour or "").upper()),
        ).fetchone()
        return int(row["review_id"])

    def pending_reviews(self, *, limit: int = 200) -> list[dict[str, Any]]:
        with self.db.connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    review_id,
                    source_name AS source,
                    source_name,
                    source_player_id,
                    raw_name,
                    normalized_name,
                    tour,
                    suggested_canonical_player_id,
                    suggested_canonical_player_id AS suggested_player_id,
                    suggested_name,
                    fuzzy_score,
                    second_score,
                    reason,
                    status,
                    created_at
                FROM alias_review_queue
                WHERE status='PENDING'
                ORDER BY created_at, review_id
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [dict(row) for row in rows]

    def resolve_review(self, review_id: int, player_id: int) -> None:
        """Resolve a queue item using a canonical ID or legacy API provider ID."""
        with self.db.connection() as connection:
            review = connection.execute(
                "SELECT * FROM alias_review_queue WHERE review_id=?",
                (int(review_id),),
            ).fetchone()
            if review is None:
                raise KeyError(f"Unknown alias review id {review_id}")
            player = connection.execute(
                "SELECT * FROM players WHERE canonical_player_id=?",
                (int(player_id),),
            ).fetchone()
            if player is None:
                mapping = connection.execute(
                    """
                    SELECT p.* FROM player_source_mappings m
                    JOIN players p ON p.canonical_player_id=m.canonical_player_id
                    WHERE m.source_name='api_tennis' AND m.source_player_id=?
                    """,
                    (str(player_id),),
                ).fetchone()
                player = mapping
            if player is None:
                raise KeyError(f"Unknown canonical or API-Tennis player id {player_id}")
            canonical_id = int(player["canonical_player_id"])
            alias_key = normalize_name(str(review["raw_name"]))
            existing = connection.execute(
                "SELECT canonical_player_id FROM player_aliases WHERE alias_name=?",
                (alias_key,),
            ).fetchone()
            if existing is not None and int(existing["canonical_player_id"]) != canonical_id:
                raise ValueError("Alias already belongs to another canonical player")
            now = utc_now()
            connection.execute(
                """
                INSERT INTO player_aliases(
                    alias_name, display_alias, canonical_player_id, source_name,
                    confidence, verified, created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?)
                ON CONFLICT(alias_name) DO UPDATE SET
                    canonical_player_id=excluded.canonical_player_id,
                    display_alias=excluded.display_alias,
                    source_name=excluded.source_name,
                    verified=1,
                    updated_at=excluded.updated_at
                """,
                (
                    alias_key,
                    str(review["raw_name"]),
                    canonical_id,
                    str(review["source_name"]),
                    100.0,
                    1,
                    now,
                    now,
                ),
            )
            source_id = str(review["source_player_id"] or "")
            if source_id:
                connection.execute(
                    """
                    INSERT INTO player_source_mappings(
                        canonical_player_id, source_name, source_player_id, raw_name,
                        verified, created_at, updated_at
                    ) VALUES (?,?,?,?,?,?,?)
                    ON CONFLICT(source_name, source_player_id) DO UPDATE SET
                        canonical_player_id=excluded.canonical_player_id,
                        raw_name=excluded.raw_name,
                        verified=1,
                        updated_at=excluded.updated_at
                    """,
                    (
                        canonical_id,
                        str(review["source_name"]),
                        source_id,
                        str(review["raw_name"]),
                        1,
                        now,
                        now,
                    ),
                )
            connection.execute(
                """
                UPDATE alias_review_queue SET
                    status='RESOLVED', resolved_at=?,
                    resolved_canonical_player_id=?
                WHERE review_id=?
                """,
                (now, canonical_id, int(review_id)),
            )

    def search_players(
        self, query: str, *, tour: str = "", limit: int = 50
    ) -> list[dict[str, Any]]:
        normalized = normalize_name(query)
        compatible = self.compatible_tours(tour)
        with self.db.connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    p.canonical_player_id,
                    p.canonical_name,
                    p.normalized_name,
                    p.dob,
                    p.nationality,
                    p.tour,
                    api.source_player_id AS api_tennis_player_id
                FROM players p
                LEFT JOIN player_source_mappings api
                  ON api.canonical_player_id=p.canonical_player_id
                 AND api.source_name='api_tennis'
                WHERE (?='' OR p.normalized_name LIKE ?)
                ORDER BY p.canonical_name
                LIMIT ?
                """,
                (normalized, f"%{normalized}%", int(limit) * 3),
            ).fetchall()
        output: list[dict[str, Any]] = []
        for row in rows:
            if compatible and str(row["tour"]) not in compatible:
                continue
            item = dict(row)
            api_id = item.get("api_tennis_player_id")
            item["provider_player_id"] = int(api_id) if str(api_id or "").isdigit() else int(item["canonical_player_id"])
            output.append(item)
            if len(output) >= int(limit):
                break
        return output

    def list_players(self, *, limit: int = 1000) -> list[dict[str, Any]]:
        return self.search_players("", limit=limit)

    def get_player(self, canonical_player_id: int) -> dict[str, Any] | None:
        with self.db.connection() as connection:
            row = connection.execute(
                "SELECT * FROM players WHERE canonical_player_id=?",
                (int(canonical_player_id),),
            ).fetchone()
        return dict(row) if row is not None else None
