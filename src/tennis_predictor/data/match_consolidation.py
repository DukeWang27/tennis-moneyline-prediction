from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tennis_predictor.data.names import normalize_name
from tennis_predictor.storage.db_manager import DatabaseManager, utc_now
from tennis_predictor.storage.identity_store import IdentityStore, normalize_source_name


_RESULT_PRIORITY = {
    "api_tennis": 500,
    "kaggle_atp_daily": 400,
    "kaggle_wta_daily": 400,
    "huge_tennis": 300,
    "sackmann": 200,
    "tennis_data": 100,
}
_SURFACE_PRIORITY = {
    "kaggle_atp_daily": 500,
    "kaggle_wta_daily": 500,
    "huge_tennis": 450,
    "api_tennis": 350,
    "sackmann": 250,
    "tennis_data": 100,
}
_ODDS_PRIORITY = {
    "kaggle_atp_daily": 500,
    "kaggle_wta_daily": 500,
    "tennis_data": 450,
}
_KNOWN_SURFACES = {"Hard", "Clay", "Grass", "Carpet"}

_SIDE_PAIRS = (
    ("player1", "player2"),
    ("player1_id", "player2_id"),
    ("player1_canonical_id", "player2_canonical_id"),
    ("player1_source_id", "player2_source_id"),
    ("player1_provider_id", "player2_provider_id"),
    ("player1_rank", "player2_rank"),
    ("player1_age", "player2_age"),
    ("player1_dob", "player2_dob"),
    ("player1_nationality", "player2_nationality"),
    ("closing_odds_player1", "closing_odds_player2"),
)


def _json_value(value: object) -> object:
    if value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _row_json(row: pd.Series) -> str:
    return json.dumps(
        {str(key): _json_value(value) for key, value in row.to_dict().items()},
        sort_keys=True,
        default=str,
    )


def _present(value: object) -> bool:
    if value is None or value is pd.NA:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip() not in {"", "nan", "None", "<NA>"}


def _normalized_round(value: object) -> str:
    return normalize_name(str(value or "unknown")) or "unknown"


def _round_order_seconds(value: object) -> int:
    text = str(value or "").strip().casefold().replace(" ", "")
    explicit = {
        "q1": 1, "q2": 2, "q3": 3, "qualifying": 3,
        "r128": 10, "128": 10, "r64": 20, "64": 20,
        "r32": 30, "32": 30, "r16": 40, "16": 40,
        "qf": 50, "quarterfinal": 50, "quarterfinals": 50,
        "sf": 60, "semifinal": 60, "semifinals": 60,
        "f": 70, "final": 70,
    }
    return explicit.get(text, 35) * 900


def match_fingerprint(
    match_date: object,
    canonical_player1_id: int,
    canonical_player2_id: int,
    tournament_name: object,
    round_name: object,
) -> str:
    """Deterministic, orientation-independent match key requested by the spec."""
    date_text = pd.Timestamp(match_date).date().isoformat()
    player_ids = sorted([int(canonical_player1_id), int(canonical_player2_id)])
    payload = "|".join(
        [
            date_text,
            str(player_ids[0]),
            str(player_ids[1]),
            normalize_name(str(tournament_name or "")),
            _normalized_round(round_name),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class ConsolidationReport:
    input_rows: int
    resolved_rows: int
    unresolved_rows: int
    unique_matches: int
    duplicate_source_rows_merged: int
    by_source: dict[str, int]
    by_segment: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def derive_competition_segment(frame: pd.DataFrame) -> pd.Series:
    tour = frame.get("tour", pd.Series("", index=frame.index)).fillna("").astype(str).str.upper()
    tier = frame.get("tier", pd.Series("", index=frame.index)).fillna("").astype(str).str.casefold()
    result = pd.Series("OTHER", index=frame.index, dtype="string")
    result = result.mask(tour.eq("ATP") & tier.eq("challenger"), "ATP_CHALLENGER")
    result = result.mask(tour.eq("ATP") & tier.eq("itf"), "ITF_MEN")
    result = result.mask(tour.eq("WTA") & tier.eq("itf"), "ITF_WOMEN")
    result = result.mask(tour.eq("ATP") & result.eq("OTHER"), "ATP_MAIN")
    result = result.mask(tour.eq("WTA") & result.eq("OTHER"), "WTA_MAIN")
    return result


class CanonicalMatchConsolidator:
    """Resolve identities, orient rows consistently, fingerprint, and merge duplicates."""

    def __init__(
        self,
        identity_store: IdentityStore,
        *,
        fuzzy_threshold: float = 97.0,
        minimum_margin: float = 4.0,
    ):
        self.identity_store = identity_store
        self.fuzzy_threshold = float(fuzzy_threshold)
        self.minimum_margin = float(minimum_margin)

        # Historical datasets contain the same players in many matches.
        # Cache both successful and unresolved identity decisions so fuzzy
        # matching runs once per unique source identity rather than once per row.
        self._resolution_cache: dict[tuple[str, str, str], Any] = {}
        self._resolution_cache_hits = 0
        self._resolution_cache_misses = 0

    def _source_id(self, row: pd.Series, side: str, source: str) -> str | None:
        candidates = (
            f"{side}_source_id",
            f"{side}_provider_id",
            f"{side}_id",
        )
        for column in candidates:
            if column in row and _present(row[column]):
                value = str(row[column]).strip()
                # Existing model IDs are not raw source IDs.
                if value.startswith(("canonical:", "api_tennis:")):
                    value = value.split(":", 1)[1]
                return value
        return None

    def _resolve_row(self, row: pd.Series) -> tuple[pd.Series | None, dict[str, Any] | None]:
        output = row.copy()
        source = normalize_source_name(str(output.get("source") or "unknown"))
        tour = str(output.get("tour") or "").upper()
        resolutions = []
        for side in ("player1", "player2"):
            source_id = self._source_id(output, side, source)
            raw_name = str(output.get(side) or "").strip()

            if source_id is not None and str(source_id).strip():
                identity_key = f"id:{str(source_id).strip()}"
            else:
                normalized_cache_name = " ".join(raw_name.casefold().split())
                identity_key = f"name:{normalized_cache_name}"

            cache_key = (source, tour, identity_key)
            resolution = self._resolution_cache.get(cache_key)

            if resolution is None:
                self._resolution_cache_misses += 1
                resolution = self.identity_store.resolve_or_create_source_player(
                    source=source,
                    source_player_id=source_id,
                    raw_name=raw_name,
                    tour=tour,
                    dob=(
                        str(output.get(f"{side}_dob"))
                        if _present(output.get(f"{side}_dob"))
                        else None
                    ),
                    nationality=(
                        str(output.get(f"{side}_nationality"))
                        if _present(output.get(f"{side}_nationality"))
                        else None
                    ),
                    fuzzy_threshold=self.fuzzy_threshold,
                    minimum_margin=self.minimum_margin,
                )
                self._resolution_cache[cache_key] = resolution
            else:
                self._resolution_cache_hits += 1

            resolutions.append(resolution)
        if not all(resolution.resolved for resolution in resolutions):
            return None, {
                **{str(key): _json_value(value) for key, value in output.to_dict().items()},
                "identity_status1": resolutions[0].status,
                "identity_status2": resolutions[1].status,
                "identity_review_id1": resolutions[0].review_id,
                "identity_review_id2": resolutions[1].review_id,
            }

        for index, side in enumerate(("player1", "player2")):
            resolution = resolutions[index]
            canonical_id = int(resolution.canonical_player_id)
            output[f"{side}_canonical_id"] = canonical_id
            output[f"{side}_id"] = self.identity_store.canonical_model_player_id(canonical_id)
            output[side] = str(resolution.canonical_name)
            output[f"identity_status{index + 1}"] = resolution.status

        output["source"] = source
        return self._orient(output), None

    @staticmethod
    def _orient(row: pd.Series) -> pd.Series:
        """Orient by canonical ID, never by winner/loser position."""
        if int(row["player1_canonical_id"]) <= int(row["player2_canonical_id"]):
            return row
        output = row.copy()
        for left, right in _SIDE_PAIRS:
            if left in output or right in output:
                left_value = output.get(left, pd.NA)
                right_value = output.get(right, pd.NA)
                output[left] = right_value
                output[right] = left_value
        if _present(output.get("player1_won")):
            output["player1_won"] = 1 - int(float(output["player1_won"]))
        return output

    def canonicalize(
        self, sources: list[tuple[pd.DataFrame, str]]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        resolved: list[pd.Series] = []
        unresolved: list[dict[str, Any]] = []

        total_rows = sum(
            len(frame)
            for frame, _ in sources
            if frame is not None and not frame.empty
        )
        processed_rows = 0

        print(
            f"[identity] Starting canonicalization of {total_rows:,} match rows",
            flush=True,
        )

        for frame, default_source in sources:
            if frame is None or frame.empty:
                continue
            for _, row in frame.iterrows():
                processed_rows += 1
                working = row.copy()
                if not _present(working.get("source")):
                    working["source"] = default_source
                canonical, rejected = self._resolve_row(working)
                if canonical is not None:
                    resolved.append(canonical)
                elif rejected is not None:
                    unresolved.append(rejected)

                if processed_rows % 5000 == 0 or processed_rows == total_rows:
                    print(
                        "[identity] "
                        f"{processed_rows:,}/{total_rows:,} rows | "
                        f"unique identities={len(self._resolution_cache):,} | "
                        f"cache hits={self._resolution_cache_hits:,} | "
                        f"cache misses={self._resolution_cache_misses:,} | "
                        f"resolved matches={len(resolved):,} | "
                        f"unresolved matches={len(unresolved):,}",
                        flush=True,
                    )

        return (
            pd.DataFrame(resolved).reset_index(drop=True),
            pd.DataFrame(unresolved).reset_index(drop=True),
        )

    @staticmethod
    def _choose_row(group: pd.DataFrame, priorities: dict[str, int]) -> pd.Series:
        ranked = group.copy()
        ranked["__priority"] = ranked["source"].map(priorities).fillna(0).astype(int)
        ranked["__completeness"] = ranked.notna().sum(axis=1)
        return ranked.sort_values(
            ["__priority", "__completeness"], ascending=[False, False]
        ).iloc[0]

    @staticmethod
    def _pick_field(
        group: pd.DataFrame,
        field: str,
        priorities: dict[str, int],
        *,
        validator=None,
    ) -> tuple[object, str | None]:
        if field not in group:
            return pd.NA, None
        ranked = group.copy()
        ranked["__priority"] = ranked["source"].map(priorities).fillna(0).astype(int)
        ranked = ranked.sort_values("__priority", ascending=False)
        for _, row in ranked.iterrows():
            value = row.get(field)
            if not _present(value):
                continue
            if validator is not None and not validator(value):
                continue
            return value, str(row["source"])
        return pd.NA, None

    def _merge_group(self, group: pd.DataFrame) -> dict[str, Any]:
        result_anchor = self._choose_row(group, _RESULT_PRIORITY)
        output = {
            str(key): _json_value(value)
            for key, value in result_anchor.drop(labels=["__priority", "__completeness"], errors="ignore").to_dict().items()
        }
        provenance: dict[str, str] = {}

        result_fields = (
            "player1_won",
            "score",
            "status",
            "retirement",
            "walkover",
            "defaulted",
            "void",
            "game_form_eligible",
            "commence_time",
            "date",
            "event_date",
        )
        for field in result_fields:
            value, source = self._pick_field(group, field, _RESULT_PRIORITY)
            if _present(value):
                output[field] = _json_value(value)
                if source:
                    provenance[field] = source

        for field in ("surface", "court", "tier", "tournament", "round", "tour"):
            validator = (
                (lambda value: str(value) in _KNOWN_SURFACES)
                if field == "surface"
                else None
            )
            value, source = self._pick_field(
                group, field, _SURFACE_PRIORITY, validator=validator
            )
            if _present(value):
                output[field] = _json_value(value)
                if source:
                    provenance[field] = source

        for field in (
            "closing_odds_player1",
            "closing_odds_player2",
            "closing_odds_bookmaker",
            "closing_odds_source",
        ):
            value, source = self._pick_field(group, field, _ODDS_PRIORITY)
            if _present(value):
                output[field] = _json_value(value)
                if source:
                    provenance[field] = source

        # Fill remaining player metadata from the most complete source without
        # allowing result/odds precedence to overwrite canonical IDs.
        for field in (
            "player1_rank",
            "player2_rank",
            "player1_age",
            "player2_age",
            "player1_dob",
            "player2_dob",
            "player1_nationality",
            "player2_nationality",
            "best_of",
            "is_qualifying",
            "match_num",
        ):
            value, source = self._pick_field(group, field, _SURFACE_PRIORITY)
            if _present(value):
                output[field] = _json_value(value)
                if source:
                    provenance[field] = source

        sources = sorted(set(group["source"].astype(str)))
        output["sources_present"] = json.dumps(sources)
        output["field_provenance"] = json.dumps(provenance, sort_keys=True)
        output["source"] = str(result_anchor["source"])
        output["surface"] = (
            str(output.get("surface"))
            if str(output.get("surface")) in _KNOWN_SURFACES
            else "Unknown"
        )
        output["surface_known"] = output["surface"] in _KNOWN_SURFACES
        output["date"] = pd.Timestamp(output["date"])
        commence = pd.to_datetime(output.get("commence_time"), errors="coerce", utc=True)
        if pd.isna(commence):
            round_seconds = _round_order_seconds(output.get("round"))
            tie_break_seconds = int(str(output["match_fingerprint"])[:8], 16) % 600
            stable_seconds = min(round_seconds + tie_break_seconds, 86_399)
            commence = pd.Timestamp(output["date"]).normalize().tz_localize("UTC") + pd.to_timedelta(
                stable_seconds, unit="s"
            )
            output["date_is_estimated"] = True
            output["chronology_precision"] = "date_round_estimated"
        else:
            output["date_is_estimated"] = bool(output.get("date_is_estimated", False))
            output["chronology_precision"] = "exact_timestamp"
        output["chronology_ts"] = commence.tz_convert(None)
        output["event_date"] = pd.to_datetime(
            output.get("event_date", output["date"]), errors="coerce"
        )
        output["match_id"] = "canonical:" + str(output["match_fingerprint"])
        output["player1_provider_id"] = output.get("player1_canonical_id")
        output["player2_provider_id"] = output.get("player2_canonical_id")
        return output

    def consolidate(
        self,
        sources: list[tuple[pd.DataFrame, str]],
        *,
        start_date: object | None = None,
        end_date: object | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, ConsolidationReport, pd.DataFrame]:
        input_rows = sum(len(frame) for frame, _ in sources if frame is not None)
        canonical, unresolved = self.canonicalize(sources)
        if canonical.empty:
            return canonical, unresolved, ConsolidationReport(
                input_rows, 0, len(unresolved), 0, 0, {}, {}
            ), pd.DataFrame()

        canonical["date"] = pd.to_datetime(canonical["date"], errors="coerce")
        canonical = canonical.dropna(subset=["date", "player1_canonical_id", "player2_canonical_id"])
        if start_date is not None:
            canonical = canonical[canonical["date"] >= pd.Timestamp(start_date)]
        if end_date is not None:
            canonical = canonical[canonical["date"] <= pd.Timestamp(end_date)]
        canonical["match_fingerprint"] = canonical.apply(
            lambda row: match_fingerprint(
                row["date"],
                int(row["player1_canonical_id"]),
                int(row["player2_canonical_id"]),
                row.get("tournament"),
                row.get("round"),
            ),
            axis=1,
        )

        merged = [self._merge_group(group) for _, group in canonical.groupby("match_fingerprint", sort=False)]
        output = pd.DataFrame(merged)
        output["competition_segment"] = derive_competition_segment(output)
        output = output.sort_values(["chronology_ts", "match_id"]).reset_index(drop=True)

        source_records = canonical[[
            "source",
            "source_match_id",
            "match_fingerprint",
        ]].copy()
        source_records["raw_record_json"] = [
            _row_json(row) for _, row in canonical.iterrows()
        ]
        source_records = source_records.rename(
            columns={"source": "source_name"}
        )
        fallback_source_ids = (
            canonical["source"].astype(str) + ":" + canonical.index.astype(str)
        )
        source_records["source_match_id"] = source_records["source_match_id"].where(
            source_records["source_match_id"].notna()
            & source_records["source_match_id"].astype(str).str.strip().ne(""),
            fallback_source_ids,
        ).astype(str)

        report = ConsolidationReport(
            input_rows=input_rows,
            resolved_rows=len(canonical),
            unresolved_rows=len(unresolved),
            unique_matches=len(output),
            duplicate_source_rows_merged=len(canonical) - len(output),
            by_source=canonical["source"].value_counts().astype(int).to_dict(),
            by_segment=output["competition_segment"].value_counts().astype(int).to_dict(),
        )
        return output, unresolved, report, source_records

    @staticmethod
    def _atomic_parquet(path: str | Path, frame: pd.DataFrame) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, destination)

    def persist(
        self,
        matches: pd.DataFrame,
        source_records: pd.DataFrame,
        *,
        database_path: str | Path,
        parquet_path: str | Path,
    ) -> None:
        """Persist candidate data without touching production model artifacts."""
        self._atomic_parquet(parquet_path, matches)
        db = DatabaseManager(database_path)
        db.initialize_core_schema()

        # canonical_matches references players inside the candidate database.
        # Copy the canonical player registry from identity.db before inserting
        # any candidate matches.
        with self.identity_store.db.connection() as identity_connection:
            identity_players = identity_connection.execute(
                """
                SELECT canonical_player_id, canonical_name, normalized_name,
                       dob, nationality, tour, created_at, updated_at
                FROM players
                """
            ).fetchall()

        player_values = [
            (
                int(player["canonical_player_id"]),
                str(player["canonical_name"]),
                str(player["normalized_name"]),
                player["dob"],
                player["nationality"],
                str(player["tour"]),
                str(player["created_at"]),
                str(player["updated_at"]),
            )
            for player in identity_players
        ]

        with db.connection() as connection:
            connection.executemany(
                """
                INSERT INTO players(
                    canonical_player_id, canonical_name, normalized_name,
                    dob, nationality, tour, created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?)
                ON CONFLICT(canonical_player_id) DO UPDATE SET
                    canonical_name=excluded.canonical_name,
                    normalized_name=excluded.normalized_name,
                    dob=excluded.dob,
                    nationality=excluded.nationality,
                    tour=excluded.tour,
                    updated_at=excluded.updated_at
                """,
                player_values,
            )

            print(
                f"[persist] Copied {len(player_values):,} canonical players "
                f"into {Path(database_path)}",
                flush=True,
            )
            print(
                f"[persist] Writing {len(matches):,} canonical matches",
                flush=True,
            )

            for _, row in matches.iterrows():
                now = utc_now()
                connection.execute(
                    """
                    INSERT INTO canonical_matches(
                        match_fingerprint, date_time, match_date,
                        canonical_player1_id, canonical_player2_id,
                        tournament_name, round_name, tour, tier, surface,
                        player1_won, score, status, retirement, walkover,
                        defaulted, void, sources_present, field_provenance,
                        created_at, updated_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(match_fingerprint) DO UPDATE SET
                        date_time=excluded.date_time,
                        match_date=excluded.match_date,
                        tournament_name=excluded.tournament_name,
                        round_name=excluded.round_name,
                        tour=excluded.tour,
                        tier=excluded.tier,
                        surface=excluded.surface,
                        player1_won=excluded.player1_won,
                        score=excluded.score,
                        status=excluded.status,
                        retirement=excluded.retirement,
                        walkover=excluded.walkover,
                        defaulted=excluded.defaulted,
                        void=excluded.void,
                        sources_present=excluded.sources_present,
                        field_provenance=excluded.field_provenance,
                        updated_at=excluded.updated_at
                    """,
                    (
                        str(row["match_fingerprint"]),
                        pd.Timestamp(row["chronology_ts"]).isoformat(),
                        pd.Timestamp(row["date"]).date().isoformat(),
                        int(row["player1_canonical_id"]),
                        int(row["player2_canonical_id"]),
                        str(row.get("tournament") or "Unknown"),
                        str(row.get("round") or "Unknown"),
                        str(row.get("tour") or ""),
                        str(row.get("tier") or ""),
                        str(row.get("surface") or "Unknown"),
                        int(row["player1_won"]) if _present(row.get("player1_won")) else None,
                        str(row.get("score") or ""),
                        str(row.get("status") or ""),
                        int(bool(row.get("retirement", False))),
                        int(bool(row.get("walkover", False))),
                        int(bool(row.get("defaulted", False))),
                        int(bool(row.get("void", False))),
                        str(row.get("sources_present") or "[]"),
                        str(row.get("field_provenance") or "{}"),
                        now,
                        now,
                    ),
                )
                if _present(row.get("closing_odds_player1")) and _present(row.get("closing_odds_player2")):
                    connection.execute(
                        """
                        INSERT INTO closing_odds(
                            match_fingerprint, source_name, bookmaker,
                            player1_decimal, player2_decimal, observed_at,
                            is_closing, created_at
                        ) VALUES (?,?,?,?,?,?,?,?)
                        ON CONFLICT(match_fingerprint, source_name, bookmaker) DO UPDATE SET
                            player1_decimal=excluded.player1_decimal,
                            player2_decimal=excluded.player2_decimal,
                            observed_at=excluded.observed_at,
                            is_closing=excluded.is_closing
                        """,
                        (
                            str(row["match_fingerprint"]),
                            str(row.get("closing_odds_source") or "kaggle"),
                            str(row.get("closing_odds_bookmaker") or "consensus"),
                            float(row["closing_odds_player1"]),
                            float(row["closing_odds_player2"]),
                            pd.Timestamp(row["chronology_ts"]).isoformat(),
                            1,
                            now,
                        ),
                    )
            for _, record in source_records.iterrows():
                connection.execute(
                    """
                    INSERT INTO match_source_records(
                        source_name, source_match_id, match_fingerprint,
                        raw_record_json, imported_at
                    ) VALUES (?,?,?,?,?)
                    ON CONFLICT(source_name, source_match_id) DO UPDATE SET
                        match_fingerprint=excluded.match_fingerprint,
                        raw_record_json=excluded.raw_record_json,
                        imported_at=excluded.imported_at
                    """,
                    (
                        str(record["source_name"]),
                        str(record["source_match_id"]),
                        str(record["match_fingerprint"]),
                        str(record["raw_record_json"]),
                        utc_now(),
                    ),
                )
