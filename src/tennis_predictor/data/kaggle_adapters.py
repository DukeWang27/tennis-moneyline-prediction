from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from tennis_predictor.data.names import normalize_name


KNOWN_SURFACES = {"Hard", "Clay", "Grass", "Carpet"}


def _column_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().casefold()).strip("_")


def _column_map(frame: pd.DataFrame) -> dict[str, str]:
    return {_column_key(column): str(column) for column in frame.columns}


def _pick(columns: dict[str, str], aliases: Iterable[str], *, required: bool = False) -> str | None:
    for alias in aliases:
        key = _column_key(alias)
        if key in columns:
            return columns[key]
    if required:
        raise ValueError(f"None of the required columns were found: {list(aliases)}")
    return None


def _series(frame: pd.DataFrame, column: str | None, default: object = pd.NA) -> pd.Series:
    if column is None:
        return pd.Series(default, index=frame.index)
    return frame[column]


def _parse_match_dates(series: pd.Series) -> pd.Series:
    """Parse ISO/text dates and compact YYYYMMDD values without epoch mistakes."""
    text = series.astype("string").str.strip()
    compact = text.str.fullmatch(r"\d{8}", na=False)
    output = pd.to_datetime(text.where(~compact), errors="coerce")
    if compact.any():
        output.loc[compact] = pd.to_datetime(
            text.loc[compact], format="%Y%m%d", errors="coerce"
        )
    return output


def _clean_surface(value: object) -> str:
    text = str(value or "").strip().casefold()
    mapping = {
        "hard": "Hard",
        "hardcourt": "Hard",
        "hard court": "Hard",
        "clay": "Clay",
        "red clay": "Clay",
        "green clay": "Clay",
        "grass": "Grass",
        "carpet": "Carpet",
        "indoor carpet": "Carpet",
    }
    result = mapping.get(text, str(value or "Unknown").strip().title())
    return result if result in KNOWN_SURFACES else "Unknown"


def _tier_from_value(value: object, *, tour: str) -> str:
    text = str(value or "").strip().casefold()
    if any(token in text for token in ("challenger", "ch")) and text not in {"championship"}:
        return "Challenger"
    if any(token in text for token in ("itf", "futures", "future", "w15", "w25", "m15", "m25")):
        return "ITF"
    if text in {"c"}:
        return "Challenger"
    if text in {"f", "s"}:
        return "ITF"
    if any(token in text for token in ("grand slam", "slam", "g")):
        return "Grand Slam"
    if any(token in text for token in ("masters", "1000", "m")):
        return "Masters"
    if any(token in text for token in ("500", "a500")):
        return "500"
    if any(token in text for token in ("250", "a250")):
        return "250"
    if any(token in text for token in ("premier", "international", "wta")):
        return "WTA"
    return "Tour" if tour in {"ATP", "WTA"} else "ITF"


def _tour_from_values(default_tour: str, *values: object) -> str:
    text = " ".join(str(value or "") for value in values).casefold()
    if any(token in text for token in ("women", "woman", "female", "wta", "itf_w", "w15", "w25", "w40", "w60", "w80", "w100")):
        return "WTA"
    if any(token in text for token in ("men", "male", "atp", "itf_m", "m15", "m25")):
        return "ATP"
    return default_tour


def _status_flags(comment: object, score: object) -> dict[str, object]:
    text = f"{comment or ''} {score or ''}".casefold()
    retirement = any(token in text for token in ("retired", "retirement", " ret", "ret."))
    walkover = any(token in text for token in ("walkover", " w/o", " wo", "w.o"))
    defaulted = "default" in text
    void = any(token in text for token in ("cancelled", "canceled", "abandoned", "not played"))
    if walkover:
        status = "Walkover"
    elif defaulted:
        status = "Default"
    elif retirement:
        status = "Retired"
    elif void:
        status = "Void"
    else:
        status = "Completed"
    return {
        "status": status,
        "retirement": retirement,
        "walkover": walkover,
        "defaulted": defaulted,
        "void": void,
        "game_form_eligible": not (retirement or walkover or defaulted or void),
    }


def _decimal(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").where(lambda values: values > 1.0)


def _stable_id(*parts: object) -> str:
    payload = "|".join(str(part or "") for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _read_csv(path: Path) -> pd.DataFrame:
    errors: list[Exception] = []
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except Exception as exc:  # pragma: no cover - exercised only by malformed source files
            errors.append(exc)
    raise ValueError(f"Could not read CSV {path}: {errors[-1]}")


def read_tabular_archive(root: str | Path) -> list[tuple[str, pd.DataFrame]]:
    """Read supported tables recursively, including SQLite tables.

    The loader intentionally ignores unknown binary formats instead of guessing.
    """
    root_path = Path(root)
    tables: list[tuple[str, pd.DataFrame]] = []
    for path in sorted(root_path.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.casefold()
        try:
            if suffix == ".csv":
                tables.append((str(path.relative_to(root_path)), _read_csv(path)))
            elif suffix in {".parquet", ".pq"}:
                tables.append((str(path.relative_to(root_path)), pd.read_parquet(path)))
            elif suffix in {".feather", ".arrow"}:
                tables.append((str(path.relative_to(root_path)), pd.read_feather(path)))
            elif suffix in {".json", ".jsonl", ".ndjson"}:
                lines = suffix in {".jsonl", ".ndjson"}
                tables.append((str(path.relative_to(root_path)), pd.read_json(path, lines=lines)))
            elif suffix in {".sqlite", ".sqlite3", ".db"}:
                with sqlite3.connect(path) as connection:
                    names = [
                        str(row[0])
                        for row in connection.execute(
                            "SELECT name FROM sqlite_master WHERE type='table'"
                        ).fetchall()
                        if not str(row[0]).startswith("sqlite_")
                    ]
                    for name in names:
                        tables.append((f"{path.relative_to(root_path)}::{name}", pd.read_sql_query(f'SELECT * FROM "{name}"', connection)))
        except Exception as exc:
            raise ValueError(f"Failed reading Kaggle table {path}: {exc}") from exc
    return tables


@dataclass(frozen=True, slots=True)
class SchemaContract:
    source_name: str
    table_name: str
    semantic_columns: dict[str, str]
    raw_columns: list[str]

    @property
    def semantic_hash(self) -> str:
        payload = json.dumps(self.semantic_columns, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        output["semantic_hash"] = self.semantic_hash
        return output


@dataclass(frozen=True, slots=True)
class KaggleNormalizeReport:
    source_name: str
    tables_seen: int
    match_tables: int
    input_rows: int
    output_rows: int
    min_date: str | None
    max_date: str | None
    unresolved_date_rows: int
    contracts: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseKaggleAdapter:
    source_name: str
    default_tour: str

    DATE_ALIASES = ("date", "match_date", "tourney_date", "event_date", "start_date")
    TOURNAMENT_ALIASES = ("tournament", "tourney_name", "tournament_name", "event_name")
    WINNER_ALIASES = ("winner", "winner_name", "player_1", "player1", "home_player")
    LOSER_ALIASES = ("loser", "loser_name", "player_2", "player2", "away_player")

    def __init__(self, source_name: str, default_tour: str):
        self.source_name = source_name
        self.default_tour = default_tour

    def contract_for(self, table_name: str, frame: pd.DataFrame) -> SchemaContract | None:
        columns = _column_map(frame)
        try:
            semantic = {
                "date": _pick(columns, self.DATE_ALIASES, required=True),
                "tournament": _pick(columns, self.TOURNAMENT_ALIASES, required=True),
                "winner": _pick(columns, self.WINNER_ALIASES, required=True),
                "loser": _pick(columns, self.LOSER_ALIASES, required=True),
            }
        except ValueError:
            return None
        optional = {
            "winner_id": _pick(columns, ("winner_id", "winner_player_id", "player1_id", "p1_id")),
            "loser_id": _pick(columns, ("loser_id", "loser_player_id", "player2_id", "p2_id")),
            "surface": _pick(columns, ("surface", "court_surface")),
            "court": _pick(columns, ("court", "indoor", "indoor_outdoor")),
            "round": _pick(columns, ("round", "tournament_round", "round_name")),
            "tier": _pick(columns, ("series", "tier", "level", "tourney_level", "tournament_type")),
            "score": _pick(columns, ("score", "result", "final_score")),
            "comment": _pick(columns, ("comment", "status", "match_status")),
            "winner_rank": _pick(columns, ("wrank", "winner_rank", "winner_ranking")),
            "loser_rank": _pick(columns, ("lrank", "loser_rank", "loser_ranking")),
            "tour": _pick(columns, ("tour", "circuit", "gender", "sex")),
            "event_type": _pick(columns, ("event_type", "match_type", "discipline", "singles_doubles")),
            "match_id": _pick(columns, ("match_id", "event_key", "id")),
            "match_num": _pick(columns, ("match_num", "match_number")),
            "best_of": _pick(columns, ("best_of", "bestof", "best_of_sets")),
            "winner_dob": _pick(columns, ("winner_dob", "winner_birth_date")),
            "loser_dob": _pick(columns, ("loser_dob", "loser_birth_date")),
            "winner_nationality": _pick(columns, ("winner_ioc", "winner_country", "winner_nationality")),
            "loser_nationality": _pick(columns, ("loser_ioc", "loser_country", "loser_nationality")),
        }
        semantic.update({key: value for key, value in optional.items() if value is not None})
        return SchemaContract(
            source_name=self.source_name,
            table_name=table_name,
            semantic_columns={key: str(value) for key, value in semantic.items()},
            raw_columns=[str(column) for column in frame.columns],
        )

    def _odds_columns(self, frame: pd.DataFrame) -> tuple[str | None, str | None, str | None]:
        columns = _column_map(frame)
        pairs = (
            ("avgw", "avgl", "average"),
            ("maxw", "maxl", "maximum"),
            ("psw", "psl", "pinnacle"),
            ("b365w", "b365l", "bet365"),
            ("cbw", "cbl", "consensus"),
            ("odd_1", "odd_2", "source"),
            ("winner_odds", "loser_odds", "source"),
        )
        for winner, loser, label in pairs:
            if winner in columns and loser in columns:
                return columns[winner], columns[loser], label
        return None, None, None

    def normalize_table(
        self, table_name: str, frame: pd.DataFrame, contract: SchemaContract
    ) -> pd.DataFrame:
        semantic = contract.semantic_columns
        output = pd.DataFrame(index=frame.index)
        output["date"] = _parse_match_dates(frame[semantic["date"]])
        output["event_date"] = output["date"]
        output["commence_time"] = pd.NaT
        output["tournament"] = frame[semantic["tournament"]].fillna("Unknown").astype(str).str.strip()
        def _flip_initials(val: str) -> str:
            val = val.strip()
            if not val:
                return val
            parts = val.split()
            # If the last element ends with a dot (e.g., "J.M." or "R."), it is an initial
            if len(parts) >= 2 and parts[-1].endswith("."):
                return f"{parts[-1]} {' '.join(parts[:-1])}"
            return val

        output["player1"] = frame[semantic["winner"]].fillna("").astype(str).str.strip().apply(_flip_initials)
        output["player2"] = frame[semantic["loser"]].fillna("").astype(str).str.strip().apply(_flip_initials)
        output["player1_won"] = 1
        output["player1_source_id"] = _series(frame, semantic.get("winner_id"), pd.NA).astype("string")
        output["player2_source_id"] = _series(frame, semantic.get("loser_id"), pd.NA).astype("string")
        output["player1_dob"] = _series(frame, semantic.get("winner_dob"), pd.NA)
        output["player2_dob"] = _series(frame, semantic.get("loser_dob"), pd.NA)
        output["player1_nationality"] = _series(frame, semantic.get("winner_nationality"), pd.NA)
        output["player2_nationality"] = _series(frame, semantic.get("loser_nationality"), pd.NA)
        output["player1_rank"] = pd.to_numeric(_series(frame, semantic.get("winner_rank")), errors="coerce")
        output["player2_rank"] = pd.to_numeric(_series(frame, semantic.get("loser_rank")), errors="coerce")

        tour_values = _series(frame, semantic.get("tour"), "")
        output["tour"] = [
            _tour_from_values(self.default_tour, raw_tour, tournament)
            for raw_tour, tournament in zip(tour_values, output["tournament"], strict=False)
        ]
        tier_values = _series(frame, semantic.get("tier"), "")
        output["tier"] = [
            _tier_from_value(f"{value or ''} {tournament}", tour=tour)
            for value, tournament, tour in zip(
                tier_values, output["tournament"], output["tour"], strict=False
            )
        ]
        output["round"] = _series(frame, semantic.get("round"), "Unknown").fillna("Unknown").astype(str)
        round_text = output["round"].str.casefold().str.strip()
        output["is_qualifying"] = round_text.str.contains("qual") | round_text.str.match(r"^q\d*$")
        surface_values = _series(frame, semantic.get("surface"), "Unknown")
        output["surface"] = surface_values.map(_clean_surface)
        output["surface_known"] = output["surface"].isin(KNOWN_SURFACES)
        output["court"] = _series(frame, semantic.get("court"), pd.NA)
        output["score"] = _series(frame, semantic.get("score"), "").fillna("").astype(str)
        comments = _series(frame, semantic.get("comment"), "").fillna("").astype(str)
        flags = [
            _status_flags(comment, score)
            for comment, score in zip(comments, output["score"], strict=False)
        ]
        for key in ("status", "retirement", "walkover", "defaulted", "void", "game_form_eligible"):
            output[key] = [item[key] for item in flags]
        output["match_num"] = pd.to_numeric(
            _series(frame, semantic.get("match_num"), 0), errors="coerce"
        ).fillna(0).astype(int)
        output["best_of"] = pd.to_numeric(
            _series(frame, semantic.get("best_of"), np.nan), errors="coerce"
        )
        event_type = _series(frame, semantic.get("event_type"), "").fillna("").astype(str).str.casefold()
        name_looks_doubles = (
            output["player1"].str.contains(r"/| & ", regex=True)
            | output["player2"].str.contains(r"/| & ", regex=True)
        )
        output["is_singles"] = ~event_type.str.contains("double") & ~name_looks_doubles

        winner_odds_col, loser_odds_col, bookmaker = self._odds_columns(frame)
        output["closing_odds_player1"] = (
            _decimal(frame[winner_odds_col]) if winner_odds_col else np.nan
        )
        output["closing_odds_player2"] = (
            _decimal(frame[loser_odds_col]) if loser_odds_col else np.nan
        )
        output["closing_odds_bookmaker"] = bookmaker
        output["closing_odds_source"] = self.source_name

        raw_match_id = _series(frame, semantic.get("match_id"), pd.NA).astype("string")
        fallback_ids = [
            _stable_id(
                table_name,
                idx,
                date,
                tournament,
                player1,
                player2,
                round_name,
            )
            for idx, date, tournament, player1, player2, round_name in zip(
                frame.index,
                output["date"],
                output["tournament"],
                output["player1"],
                output["player2"],
                output["round"],
                strict=False,
            )
        ]
        output["source_match_id"] = raw_match_id.where(
            raw_match_id.notna() & raw_match_id.str.strip().ne(""),
            pd.Series(fallback_ids, index=frame.index, dtype="string"),
        )
        output["match_id"] = self.source_name + ":" + output["source_match_id"].astype(str)
        output["source"] = self.source_name
        output["source_file"] = table_name
        output["date_is_estimated"] = True
        output["raw_winner_name"] = output["player1"]
        output["raw_loser_name"] = output["player2"]
        return output

    def normalize_archive(
        self, root: str | Path
    ) -> tuple[pd.DataFrame, KaggleNormalizeReport]:
        tables = read_tabular_archive(root)
        frames: list[pd.DataFrame] = []
        contracts: list[SchemaContract] = []
        input_rows = 0
        for table_name, frame in tables:
            contract = self.contract_for(table_name, frame)
            if contract is None:
                continue
            input_rows += len(frame)
            contracts.append(contract)
            frames.append(self.normalize_table(table_name, frame, contract))
        if not contracts:
            raise ValueError(
                f"No match table in {root} satisfied the required date/tournament/winner/loser schema."
            )
        output = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        unresolved_dates = int(output["date"].isna().sum()) if not output.empty else 0
        output = output.dropna(subset=["date"])
        output = output[
            output["player1"].astype(str).str.strip().ne("")
            & output["player2"].astype(str).str.strip().ne("")
            & output["is_singles"].fillna(True)
        ].copy()
        report = KaggleNormalizeReport(
            source_name=self.source_name,
            tables_seen=len(tables),
            match_tables=len(contracts),
            input_rows=input_rows,
            output_rows=len(output),
            min_date=(output["date"].min().date().isoformat() if not output.empty else None),
            max_date=(output["date"].max().date().isoformat() if not output.empty else None),
            unresolved_date_rows=unresolved_dates,
            contracts=[contract.to_dict() for contract in contracts],
        )
        return output.reset_index(drop=True), report


class DissfyaDailyAdapter(BaseKaggleAdapter):
    pass


class HugeTennisAdapter(BaseKaggleAdapter):
    """Flexible adapter for the static archive's match-like tables.

    It accepts CSV/Parquet/SQLite layouts and only imports tables that satisfy
    the semantic contract. Player/ranking-only tables are left archived for
    future enrichment rather than being misread as matches.
    """

    pass


def adapter_for(source_name: str) -> BaseKaggleAdapter:
    if source_name == "kaggle_atp_daily":
        return DissfyaDailyAdapter(source_name, "ATP")
    if source_name == "kaggle_wta_daily":
        return DissfyaDailyAdapter(source_name, "WTA")
    if source_name == "huge_tennis":
        return HugeTennisAdapter(source_name, "ATP")
    raise KeyError(f"Unsupported Kaggle source {source_name}")
