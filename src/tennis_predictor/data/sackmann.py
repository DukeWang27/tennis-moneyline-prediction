from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from tennis_predictor.config import DataConfig

LOGGER = logging.getLogger(__name__)

BASE_URLS = {
    "ATP": "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master",
    "WTA": "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master",
}

ATP_500_PATTERN = re.compile(
    r"(rotterdam|dubai|acapulco|barcelona|hamburg|washington|beijing|tokyo|basel|vienna|"
    r"rio de janeiro|halle|queen'?s|stuttgart|mexican open|china open|erste bank|swiss indoors|"
    r"german open|memphis|valencia)",
    re.IGNORECASE,
)

ROUND_OFFSET_REGULAR = {
    "Q1": -2,
    "Q2": -1,
    "Q3": 0,
    "R128": 0,
    "R64": 1,
    "R32": 2,
    "R16": 3,
    "QF": 4,
    "SF": 5,
    "F": 6,
    "RR": 1,
    "BR": 6,
}
ROUND_OFFSET_SLAM = {
    "Q1": -5,
    "Q2": -3,
    "Q3": -1,
    "R128": 0,
    "R64": 2,
    "R32": 4,
    "R16": 7,
    "QF": 9,
    "SF": 11,
    "F": 13,
}


def _safe_text(series: pd.Series, default: str = "") -> pd.Series:
    if series is None:
        return pd.Series(dtype="string")
    return series.astype("string").fillna(default)


def _score_flags(score: str | None) -> tuple[bool, bool, bool]:
    text = str(score or "").upper().strip()
    walkover = any(token in text for token in ("W/O", "WO", "WALKOVER"))
    retirement = any(token in text for token in ("RET", "ABD", "ABN"))
    defaulted = "DEF" in text
    incomplete = retirement or walkover or defaulted
    return retirement, walkover, incomplete


def _stable_flip(match_id: str) -> bool:
    digest = hashlib.blake2b(match_id.encode("utf-8"), digest_size=1).digest()[0]
    return bool(digest % 2)


def _estimate_match_date(event_date: pd.Timestamp, round_name: str, tier: str) -> pd.Timestamp:
    offsets = ROUND_OFFSET_SLAM if tier == "Slam" else ROUND_OFFSET_REGULAR
    return event_date + pd.Timedelta(days=offsets.get(str(round_name), 0))


class SackmannLoader:
    """Download and standardize Jeff Sackmann ATP/WTA match files."""

    def __init__(self, config: DataConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.tier_overrides = self._load_tier_overrides(config.tier_overrides_path)

    @staticmethod
    def _load_tier_overrides(path: str | None) -> dict[tuple[str, str], str]:
        if not path or not Path(path).exists():
            return {}
        frame = pd.read_csv(path)
        required = {"tour", "tournament", "tier"}
        if not required.issubset(frame.columns):
            raise ValueError(f"Tier override file must contain {sorted(required)}")
        return {
            (str(row.tour).upper(), str(row.tournament).casefold().strip()): str(row.tier)
            for row in frame.itertuples(index=False)
        }

    def _remote_names(self, tour: str, year: int) -> list[str]:
        prefix = tour.lower()
        names = [f"{prefix}_matches_{year}.csv"]
        if self.config.include_lower_levels:
            if tour == "ATP":
                names.append(f"atp_matches_qual_chall_{year}.csv")
            else:
                names.append(f"wta_matches_qual_itf_{year}.csv")
        return names

    def _download_one(self, tour: str, filename: str, force: bool = False) -> Path | None:
        target = self.cache_dir / tour.lower() / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size > 50 and not force:
            return target
        url = f"{BASE_URLS[tour]}/{filename}"
        response = requests.get(url, timeout=self.config.request_timeout_seconds)
        if response.status_code == 404:
            LOGGER.info("No source file at %s", url)
            return None
        response.raise_for_status()
        target.write_bytes(response.content)
        return target

    def download(self, force: bool = False) -> list[Path]:
        paths: list[Path] = []
        for tour in ("ATP", "WTA"):
            for year in range(self.config.start_year, self.config.end_year + 1):
                for filename in self._remote_names(tour, year):
                    path = self._download_one(tour, filename, force=force)
                    if path is not None:
                        paths.append(path)
        return paths

    def load(self, download: bool = True, force_download: bool = False) -> pd.DataFrame:
        if download:
            paths = self.download(force=force_download)
        else:
            paths = sorted(self.cache_dir.glob("**/*_matches_*.csv"))
        frames: list[pd.DataFrame] = []
        for path in paths:
            tour = "ATP" if path.name.startswith("atp_") else "WTA"
            try:
                raw = pd.read_csv(path, low_memory=False)
                frames.append(self._standardize(raw, tour=tour, source_file=path.name))
            except Exception as exc:  # keep long downloads auditable
                raise RuntimeError(f"Failed to parse {path}: {exc}") from exc
        if not frames:
            raise FileNotFoundError("No Sackmann match files were found or downloaded.")
        matches = pd.concat(frames, ignore_index=True, sort=False)
        matches = matches.drop_duplicates("match_id", keep="last")
        matches = self._apply_exact_dates(matches)
        return matches.sort_values(["date", "event_date", "match_num", "match_id"]).reset_index(drop=True)

    def _tier(self, tour: str, level: str, tournament: str, source_file: str) -> str:
        override = self.tier_overrides.get((tour, tournament.casefold().strip()))
        if override:
            return override
        level = (level or "").upper()
        if level == "G":
            return "Slam"
        if level in {"M", "P", "PM"}:
            return "Masters"
        if level == "C" or "chall" in source_file:
            return "Challenger"
        if level in {"I", "F"} or "itf" in source_file:
            return "ITF"
        if level in {"D", "B"}:
            return "Team"
        if level == "A":
            return "500" if ATP_500_PATTERN.search(tournament) else "250"
        return "Other"

    def _standardize(self, raw: pd.DataFrame, tour: str, source_file: str) -> pd.DataFrame:
        required = {
            "tourney_id",
            "tourney_name",
            "surface",
            "tourney_level",
            "tourney_date",
            "match_num",
            "winner_name",
            "loser_name",
            "score",
            "round",
        }
        missing = required - set(raw.columns)
        if missing:
            raise ValueError(f"{source_file} is missing columns: {sorted(missing)}")

        df = raw.copy()
        event_date = pd.to_datetime(
            df["tourney_date"].astype("Int64").astype("string"),
            format="%Y%m%d",
            errors="coerce",
        )
        match_num = pd.to_numeric(df["match_num"], errors="coerce").fillna(-1).astype(int)
        match_id = (
            tour
            + "_"
            + _safe_text(df["tourney_id"], "unknown")
            + "_"
            + match_num.astype(str)
        )
        tournament = _safe_text(df["tourney_name"], "Unknown")
        level = _safe_text(df["tourney_level"], "")
        tier = pd.Series(
            [
                self._tier(tour, lvl, name, source_file)
                for lvl, name in zip(level, tournament, strict=False)
            ],
            index=df.index,
            dtype="string",
        )

        score = _safe_text(df["score"], "")
        flags = score.map(_score_flags)
        retirement = flags.map(lambda x: x[0]).astype(bool)
        walkover = flags.map(lambda x: x[1]).astype(bool)
        incomplete = flags.map(lambda x: x[2]).astype(bool)

        winner_id = self._ids(df.get("winner_id"), df["winner_name"], tour)
        loser_id = self._ids(df.get("loser_id"), df["loser_name"], tour)
        flips = match_id.map(_stable_flip)

        def choose(winner_values: Iterable, loser_values: Iterable, first: bool) -> list:
            output = []
            for flip, winner, loser in zip(flips, winner_values, loser_values, strict=False):
                output.append(loser if (flip == first) else winner)
            return output

        # With first=True: flipped rows use loser as player1. With first=False:
        # flipped rows use winner as player2.
        p1_name = choose(df["winner_name"], df["loser_name"], True)
        p2_name = choose(df["winner_name"], df["loser_name"], False)
        p1_id = choose(winner_id, loser_id, True)
        p2_id = choose(winner_id, loser_id, False)
        p1_age = choose(df.get("winner_age", np.nan), df.get("loser_age", np.nan), True)
        p2_age = choose(df.get("winner_age", np.nan), df.get("loser_age", np.nan), False)
        p1_rank = choose(df.get("winner_rank", np.nan), df.get("loser_rank", np.nan), True)
        p2_rank = choose(df.get("winner_rank", np.nan), df.get("loser_rank", np.nan), False)
        p1_won = (~flips).astype(int)

        dates = [
            _estimate_match_date(date, rnd, tr)
            for date, rnd, tr in zip(event_date, df["round"], tier, strict=False)
        ]

        clean = pd.DataFrame(
            {
                "match_id": match_id,
                "tour": tour,
                "date": pd.to_datetime(dates),
                "event_date": event_date,
                "date_is_estimated": True,
                "tournament": tournament,
                "tournament_level_raw": level,
                "tier": tier,
                "round": _safe_text(df["round"], ""),
                "surface": _safe_text(df["surface"], "Unknown").str.title(),
                "match_num": match_num,
                "player1": p1_name,
                "player2": p2_name,
                "player1_id": p1_id,
                "player2_id": p2_id,
                "player1_age": pd.to_numeric(pd.Series(p1_age), errors="coerce"),
                "player2_age": pd.to_numeric(pd.Series(p2_age), errors="coerce"),
                "player1_rank": pd.to_numeric(pd.Series(p1_rank), errors="coerce"),
                "player2_rank": pd.to_numeric(pd.Series(p2_rank), errors="coerce"),
                "winner": _safe_text(df["winner_name"], ""),
                "winner_id": winner_id,
                "loser_id": loser_id,
                "player1_won": p1_won,
                "score": score,
                "retirement": retirement,
                "walkover": walkover,
                "incomplete_match": incomplete,
                "retiring_player_id": np.where(retirement | walkover, loser_id, pd.NA),
                "best_of": pd.to_numeric(df.get("best_of", np.nan), errors="coerce"),
                "source_file": source_file,
            }
        )
        clean = clean.dropna(subset=["date", "player1_id", "player2_id"])
        clean = clean[clean["player1_id"] != clean["player2_id"]]
        return clean

    @staticmethod
    def _ids(id_series: pd.Series | None, names: pd.Series, tour: str) -> pd.Series:
        if id_series is None:
            numeric = pd.Series(pd.NA, index=names.index, dtype="string")
        else:
            numeric = pd.to_numeric(id_series, errors="coerce").astype("Int64").astype("string")
        fallback = names.astype("string").fillna("").map(
            lambda name: hashlib.blake2b(name.casefold().encode(), digest_size=8).hexdigest()
        )
        values = numeric.where(numeric.notna() & (numeric != "<NA>"), "name-" + fallback)
        return tour + ":" + values

    def _apply_exact_dates(self, matches: pd.DataFrame) -> pd.DataFrame:
        path = self.config.exact_dates_path
        if not path or not Path(path).exists():
            return matches
        exact = pd.read_csv(path)
        if not {"match_id", "date"}.issubset(exact.columns):
            raise ValueError("Exact-date file must have match_id,date columns")
        exact["date"] = pd.to_datetime(exact["date"], errors="coerce")
        mapping = exact.dropna(subset=["date"]).set_index("match_id")["date"]
        mask = matches["match_id"].isin(mapping.index)
        matches.loc[mask, "date"] = matches.loc[mask, "match_id"].map(mapping)
        matches.loc[mask, "date_is_estimated"] = False
        return matches
