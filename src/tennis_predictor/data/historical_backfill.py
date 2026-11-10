from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd
import requests

from tennis_predictor.data.names import normalize_name
from tennis_predictor.storage.identity_store import IdentityStore


_VALID_SURFACES = {"Hard", "Clay", "Grass", "Carpet"}
_WINNER_FIRST = {"first player", "first", "1"}
_WINNER_SECOND = {"second player", "second", "2"}


def _truthy(value: object) -> bool:
    return str(value or "").casefold().strip() in {"1", "true", "yes", "y"}


def _stable_flip(value: str) -> bool:
    return bool(hashlib.blake2b(value.encode("utf-8"), digest_size=1).digest()[0] % 2)


def _normalize_status(value: object) -> str:
    return normalize_name(str(value or ""))


def _surface(value: object) -> str | None:
    text = str(value or "").strip().title()
    return text if text in _VALID_SURFACES else None


def _score_text(item: dict[str, Any]) -> str:
    scores = item.get("scores") or []
    parts: list[str] = []
    if isinstance(scores, list):
        for score in scores:
            if not isinstance(score, dict):
                continue
            left = score.get("score_first")
            right = score.get("score_second")
            if left is not None and right is not None:
                parts.append(f"{left}-{right}")
    return " ".join(parts) or str(item.get("event_final_result") or "")


def _clean_round(value: object) -> str:
    text = normalize_name(str(value or ""))
    mappings = [
        ("qualification final", "QFQ"),
        ("qualification 3", "Q3"),
        ("qualification 2", "Q2"),
        ("qualification 1", "Q1"),
        ("round of 128", "R128"),
        ("round of 64", "R64"),
        ("round of 32", "R32"),
        ("round of 16", "R16"),
        ("1 128 finals", "R128"),
        ("1 64 finals", "R64"),
        ("1 32 finals", "R32"),
        ("1 16 finals", "R16"),
        ("quarter", "QF"),
        ("semi", "SF"),
        ("final", "F"),
        ("round robin", "RR"),
    ]
    for needle, code in mappings:
        if needle in text:
            return code
    return str(value or "")


def classify_event(event_type: object, tournament_name: object) -> tuple[str, str, str] | None:
    """Return (tour, tier, competition_segment) for supported singles events."""
    event = normalize_name(str(event_type or ""))
    tournament = normalize_name(str(tournament_name or ""))
    if not event or any(token in event for token in ("doubles", "boys", "girls", "mixed")):
        return None
    if "exhibition" in event:
        return None

    if "itf men" in event and "singles" in event:
        return "ATP", "ITF", "ITF_MEN"
    if "itf women" in event and "singles" in event:
        return "WTA", "ITF", "ITF_WOMEN"
    if "challenger men" in event and "singles" in event:
        return "ATP", "Challenger", "ATP_CHALLENGER"
    if "challenger women" in event and "singles" in event:
        return "WTA", "125", "WTA_MAIN"
    if "atp" in event and "singles" in event:
        if any(name in tournament for name in ("australian open", "roland garros", "french open", "wimbledon", "us open")):
            tier = "Slam"
        elif "masters" in tournament:
            tier = "Masters"
        else:
            tier = "Other"
        return "ATP", tier, "ATP_MAIN"
    if "wta" in event and "singles" in event:
        if any(name in tournament for name in ("australian open", "roland garros", "french open", "wimbledon", "us open")):
            tier = "Slam"
        else:
            tier = "Other"
        return "WTA", tier, "WTA_MAIN"
    return None


def iter_seven_day_windows(start: object, end: object) -> Iterator[tuple[pd.Timestamp, pd.Timestamp]]:
    cursor = pd.Timestamp(start).normalize()
    stop = pd.Timestamp(end).normalize()
    if stop < cursor:
        raise ValueError("end date must be on or after start date")
    while cursor <= stop:
        window_end = min(cursor + pd.Timedelta(days=6), stop)
        yield cursor, window_end
        cursor = window_end + pd.Timedelta(days=1)


def raw_window_path(root: str | Path, start: pd.Timestamp, end: pd.Timestamp) -> Path:
    directory = Path(root) / f"{start.year:04d}" / f"{start.month:02d}"
    return directory / f"{start.date().isoformat()}_{end.date().isoformat()}.json"


@dataclass(slots=True)
class HistoricalDownloadReport:
    start_date: str
    end_date: str
    requested_windows: int = 0
    downloaded_windows: int = 0
    skipped_windows: int = 0
    failed_windows: int = 0
    raw_files: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["raw_files"] = self.raw_files or []
        return data


class ApiTennisRawDownloader:
    """Download API-Tennis fixtures and archive each raw response before parsing."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_key_env: str = "API_TENNIS_KEY",
        base_url: str = "https://api.api-tennis.com/tennis/",
        timeout_seconds: int = 45,
        max_retries: int = 4,
        retry_backoff_seconds: float = 2.0,
        sleep_between_requests: float = 0.25,
        session: requests.Session | None = None,
    ):
        self.api_key = api_key
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.timeout_seconds = int(timeout_seconds)
        self.max_retries = int(max_retries)
        self.retry_backoff_seconds = float(retry_backoff_seconds)
        self.sleep_between_requests = float(sleep_between_requests)
        self.session = session or requests.Session()

    def _key(self) -> str:
        key = self.api_key or os.getenv(self.api_key_env)
        if not key:
            raise RuntimeError(f"Set {self.api_key_env} before running the historical backfill.")
        return key

    @staticmethod
    def _atomic_json_write(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)

    def _fetch_window(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        *,
        timezone_name: str,
    ) -> dict[str, Any]:
        params = {
            "method": "get_fixtures",
            "APIkey": self._key(),
            "date_start": start.date().isoformat(),
            "date_stop": end.date().isoformat(),
            "timezone": timezone_name,
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(
                    self.base_url,
                    params=params,
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise RuntimeError("API-Tennis returned a non-object JSON payload")
                return payload
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_backoff_seconds * (2**attempt))
        raise RuntimeError(
            f"API-Tennis failed for {start.date()} through {end.date()}: {last_error}"
        ) from last_error

    def download(
        self,
        *,
        start_date: object,
        end_date: object | None,
        raw_root: str | Path,
        timezone_name: str = "America/New_York",
        force: bool = False,
        continue_on_error: bool = False,
    ) -> HistoricalDownloadReport:
        start = pd.Timestamp(start_date).normalize()
        end = pd.Timestamp(end_date or date.today()).normalize()
        report = HistoricalDownloadReport(
            start_date=start.date().isoformat(),
            end_date=end.date().isoformat(),
            raw_files=[],
        )
        for window_start, window_end in iter_seven_day_windows(start, end):
            report.requested_windows += 1
            output = raw_window_path(raw_root, window_start, window_end)
            if output.exists() and not force:
                report.skipped_windows += 1
                report.raw_files.append(str(output))
                continue
            try:
                payload = self._fetch_window(
                    window_start,
                    window_end,
                    timezone_name=timezone_name,
                )
                # Archive the exact response before validating or parsing it.
                self._atomic_json_write(output, payload)
                report.raw_files.append(str(output))
                if int(payload.get("success", 0)) != 1:
                    raise RuntimeError(f"API-Tennis error payload archived at {output}: {payload}")
                report.downloaded_windows += 1
            except Exception:
                report.failed_windows += 1
                if not continue_on_error:
                    raise
            if self.sleep_between_requests > 0:
                time.sleep(self.sleep_between_requests)
        return report


@dataclass(slots=True)
class HistoricalParseReport:
    raw_files: int
    raw_events: int
    normalized_matches: int
    unresolved_rows: int
    skipped_non_singles: int
    duplicate_matches: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class ApiTennisHistoryParser:
    def __init__(self, identity_store: IdentityStore):
        self.identity_store = identity_store

    @staticmethod
    def raw_files(raw_root: str | Path) -> list[Path]:
        return sorted(Path(raw_root).glob("**/*.json"))

    @staticmethod
    def load_payload(path: str | Path) -> dict[str, Any]:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Raw API-Tennis file is not a JSON object: {path}")
        return payload

    def _player(
        self,
        *,
        provider_id: object,
        name: str,
        player_tour: str,
    ) -> tuple[int, str]:
        if provider_id is None or not str(provider_id).strip().isdigit():
            raise ValueError("missing numeric API-Tennis player key")
        player_id = int(str(provider_id).strip())
        self.identity_store.upsert_player(
            player_id,
            canonical_name=name,
            tour=player_tour,
            aliases=[name],
        )
        self.identity_store.link_source_id(
            source="api_tennis",
            source_player_id=player_id,
            provider_player_id=player_id,
        )
        return player_id, self.identity_store.model_player_id(player_id)

    @staticmethod
    def _status_flags(item: dict[str, Any]) -> dict[str, bool]:
        status = _normalize_status(item.get("event_status"))
        final_result = _normalize_status(item.get("event_final_result"))
        retirement = "retir" in status or "retir" in final_result or status == "ret"
        walkover = "walkover" in status or status in {"wo", "w o"} or "walkover" in final_result
        defaulted = "default" in status or "default" in final_result
        cancelled = status in {"cancelled", "canceled"}
        abandoned = "abandon" in status
        return {
            "retirement": retirement,
            "walkover": walkover,
            "defaulted": defaulted,
            "cancelled": cancelled,
            "abandoned": abandoned,
            "void": cancelled or abandoned,
        }

    def parse_item(self, item: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        tournament = str(item.get("tournament_name") or "Unknown").strip()
        classification = classify_event(item.get("event_type_type"), tournament)
        if classification is None:
            return None, "unsupported or non-singles event"
        tour, tier, segment = classification
        identity_tour = "ITF_M" if segment == "ITF_MEN" else "ITF_W" if segment == "ITF_WOMEN" else tour

        event_key = str(item.get("event_key") or "").strip()
        if not event_key:
            return None, "missing event_key"
        first_name = str(item.get("event_first_player") or "").strip()
        second_name = str(item.get("event_second_player") or "").strip()
        if not first_name or not second_name:
            return None, "missing player name"

        winner = _normalize_status(item.get("event_winner"))
        if winner in _WINNER_FIRST:
            first_won = 1.0
        elif winner in _WINNER_SECOND:
            first_won = 0.0
        else:
            return None, "completed/trainable fixture has no winner"

        event_date = pd.to_datetime(item.get("event_date"), errors="coerce")
        if pd.isna(event_date):
            return None, "invalid event_date"
        event_time = str(item.get("event_time") or "00:00").strip() or "00:00"
        chronology = pd.to_datetime(
            f"{pd.Timestamp(event_date).date().isoformat()} {event_time}",
            errors="coerce",
        )
        if pd.isna(chronology):
            chronology = pd.Timestamp(event_date).normalize()

        try:
            first_provider_id, first_model_id = self._player(
                provider_id=item.get("first_player_key"),
                name=first_name,
                player_tour=identity_tour,
            )
            second_provider_id, second_model_id = self._player(
                provider_id=item.get("second_player_key"),
                name=second_name,
                player_tour=identity_tour,
            )
        except ValueError as exc:
            return None, str(exc)

        tournament_id = str(item.get("tournament_key") or "").strip()
        explicit_surface = (
            _surface(item.get("event_surface"))
            or _surface(item.get("surface"))
            or _surface(item.get("tournament_surface"))
        )
        remembered = self.identity_store.resolve_tournament(
            provider="api_tennis",
            provider_tournament_id=tournament_id,
            tournament_name=tournament,
            tour=tour,
        )
        surface = explicit_surface or (remembered or {}).get("surface") or "Unknown"
        resolved_tier = (remembered or {}).get("tier") or tier
        self.identity_store.upsert_tournament(
            provider="api_tennis",
            provider_tournament_id=tournament_id or normalize_name(tournament),
            canonical_name=tournament,
            tour=tour,
            surface=surface,
            tier=resolved_tier,
        )

        flags = self._status_flags(item)
        is_qualifying = _truthy(item.get("event_qualification")) or "qualif" in normalize_name(
            str(item.get("tournament_round") or "")
        )

        match_id = f"api_tennis:{event_key}"
        player1_name, player2_name = first_name, second_name
        player1_provider_id, player2_provider_id = first_provider_id, second_provider_id
        player1_id, player2_id = first_model_id, second_model_id
        player1_won = first_won
        if _stable_flip(match_id):
            player1_name, player2_name = player2_name, player1_name
            player1_provider_id, player2_provider_id = player2_provider_id, player1_provider_id
            player1_id, player2_id = player2_id, player1_id
            player1_won = 1.0 - first_won

        row = {
            "fixture_id": match_id,
            "match_id": match_id,
            "provider_event_id": event_key,
            "source": "api_tennis",
            "source_file": "api_tennis_archive",
            "tour": tour,
            "competition_segment": segment,
            "date": pd.Timestamp(event_date).normalize(),
            "event_date": pd.Timestamp(event_date).normalize(),
            "commence_time": pd.Timestamp(chronology),
            "chronology_ts": pd.Timestamp(chronology),
            "date_is_estimated": False,
            "tournament": tournament,
            "tournament_key": tournament_id,
            "tournament_level_raw": str(item.get("event_type_type") or ""),
            "tier": resolved_tier,
            "round": _clean_round(item.get("tournament_round")),
            "is_qualifying": bool(is_qualifying),
            "surface": surface if surface in _VALID_SURFACES else "Unknown",
            "surface_known": surface in _VALID_SURFACES,
            "match_num": int(event_key) if event_key.isdigit() else 0,
            "player1": player1_name,
            "player2": player2_name,
            "player1_provider_id": int(player1_provider_id),
            "player2_provider_id": int(player2_provider_id),
            "player1_id": player1_id,
            "player2_id": player2_id,
            "player1_age": np.nan,
            "player2_age": np.nan,
            "player1_rank": np.nan,
            "player2_rank": np.nan,
            "winner": player1_name if player1_won == 1 else player2_name,
            "winner_id": player1_id if player1_won == 1 else player2_id,
            "loser_id": player2_id if player1_won == 1 else player1_id,
            "player1_won": float(player1_won),
            "score": _score_text(item),
            "event_status": str(item.get("event_status") or ""),
            **flags,
            "incomplete_match": bool(
                flags["retirement"] or flags["walkover"] or flags["defaulted"] or flags["void"]
            ),
            "game_form_eligible": not bool(
                flags["retirement"] or flags["walkover"] or flags["defaulted"] or flags["void"]
            ),
            "retiring_player_id": (
                player2_id if player1_won == 1 else player1_id
            ) if flags["retirement"] else pd.NA,
            "best_of": np.nan,
        }
        return row, None

    def parse_archive(
        self,
        raw_root: str | Path,
    ) -> tuple[pd.DataFrame, pd.DataFrame, HistoricalParseReport]:
        rows: list[dict[str, Any]] = []
        unresolved: list[dict[str, Any]] = []
        raw_events = 0
        skipped_non_singles = 0
        files = self.raw_files(raw_root)
        for path in files:
            payload = self.load_payload(path)
            if int(payload.get("success", 0)) != 1:
                unresolved.append(
                    {
                        "raw_file": str(path),
                        "unresolved_reason": "archived API error payload",
                        "payload": json.dumps(payload, default=str),
                    }
                )
                continue
            result = payload.get("result", [])
            if not isinstance(result, list):
                unresolved.append(
                    {
                        "raw_file": str(path),
                        "unresolved_reason": "result is not a list",
                    }
                )
                continue
            for item in result:
                if not isinstance(item, dict):
                    continue
                raw_events += 1
                row, reason = self.parse_item(item)
                if row is None:
                    if reason == "unsupported or non-singles event":
                        skipped_non_singles += 1
                    else:
                        unresolved.append(
                            {
                                "raw_file": str(path),
                                "event_key": item.get("event_key"),
                                "event_date": item.get("event_date"),
                                "event_type_type": item.get("event_type_type"),
                                "tournament_name": item.get("tournament_name"),
                                "event_first_player": item.get("event_first_player"),
                                "event_second_player": item.get("event_second_player"),
                                "unresolved_reason": reason,
                            }
                        )
                    continue
                row["raw_file"] = str(path)
                rows.append(row)

        normalized = pd.DataFrame(rows)
        duplicate_matches = 0
        if not normalized.empty:
            before = len(normalized)
            normalized = (
                normalized.sort_values(["chronology_ts", "match_id"])
                .drop_duplicates("match_id", keep="last")
                .reset_index(drop=True)
            )
            duplicate_matches = before - len(normalized)
        unresolved_frame = pd.DataFrame(unresolved)
        report = HistoricalParseReport(
            raw_files=len(files),
            raw_events=raw_events,
            normalized_matches=len(normalized),
            unresolved_rows=len(unresolved_frame),
            skipped_non_singles=skipped_non_singles,
            duplicate_matches=duplicate_matches,
        )
        return normalized, unresolved_frame, report

    @staticmethod
    def save(
        matches: pd.DataFrame,
        unresolved: pd.DataFrame,
        *,
        normalized_path: str | Path,
        unresolved_path: str | Path,
    ) -> None:
        normalized_output = Path(normalized_path)
        unresolved_output = Path(unresolved_path)
        normalized_output.parent.mkdir(parents=True, exist_ok=True)
        unresolved_output.parent.mkdir(parents=True, exist_ok=True)
        matches.to_parquet(normalized_output, index=False)
        unresolved.to_csv(unresolved_output, index=False)


class ApiTennisProfileHydrator:
    """Optional quota-aware player-profile hydration for DOB/nationality fields."""

    def __init__(
        self,
        identity_store: IdentityStore,
        *,
        raw_root: str | Path,
        api_key: str | None = None,
        api_key_env: str = "API_TENNIS_KEY",
        base_url: str = "https://api.api-tennis.com/tennis/",
        timeout_seconds: int = 45,
    ):
        self.identity_store = identity_store
        self.raw_root = Path(raw_root) / "players"
        self.api_key = api_key
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.timeout_seconds = int(timeout_seconds)

    def _key(self) -> str:
        key = self.api_key or os.getenv(self.api_key_env)
        if not key:
            raise RuntimeError(f"Set {self.api_key_env} before hydrating player profiles.")
        return key

    def hydrate(self, player_ids: Iterable[int], *, max_profiles: int | None = None) -> int:
        count = 0
        for player_id in sorted({int(value) for value in player_ids}):
            if max_profiles is not None and count >= int(max_profiles):
                break
            output = self.raw_root / f"{player_id}.json"
            if output.exists():
                with output.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            else:
                response = requests.get(
                    self.base_url,
                    params={
                        "method": "get_players",
                        "APIkey": self._key(),
                        "player_key": player_id,
                    },
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                payload = response.json()
                ApiTennisRawDownloader._atomic_json_write(output, payload)
            result = payload.get("result", []) if isinstance(payload, dict) else []
            if not isinstance(result, list) or not result:
                continue
            profile = result[0]
            name = str(profile.get("player_name") or "").strip()
            if not name:
                continue
            existing = self.identity_store.resolve_source_id(
                source="api_tennis", source_player_id=player_id
            )
            tour = "" if existing.provider_player_id is None else ""
            # Upsert retains the previously known tour when this call provides blank.
            self.identity_store.upsert_player(
                player_id,
                canonical_name=name,
                tour=tour,
                dob=str(profile.get("player_bday") or "") or None,
                nationality=str(profile.get("player_country") or "") or None,
                aliases=[name],
            )
            self.identity_store.link_source_id(
                source="api_tennis",
                source_player_id=player_id,
                provider_player_id=player_id,
            )
            count += 1
        return count
