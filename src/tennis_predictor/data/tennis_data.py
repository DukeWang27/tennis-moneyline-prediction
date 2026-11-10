from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


_ODDS_PAIRS = {
    "bet365": ("B365W", "B365L"),
    "pinnacle": ("PSW", "PSL"),
    "market_max": ("MaxW", "MaxL"),
    "market_average": ("AvgW", "AvgL"),
    "centrebet": ("CBW", "CBL"),
    "expekt": ("EXW", "EXL"),
    "ladbrokes": ("LBW", "LBL"),
    "stan_james": ("SJW", "SJL"),
    "unibet": ("UBW", "UBL"),
}


@dataclass(slots=True)
class TennisDataLoadReport:
    files: int
    source_rows: int
    odds_rows: int
    skipped_files: int


class TennisDataLoader:
    """Normalize local Tennis-Data CSV files into the existing odds-snapshot schema."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    @staticmethod
    def _read(path: Path) -> pd.DataFrame:
        errors: list[Exception] = []
        for encoding in ("utf-8", "latin-1"):
            try:
                return pd.read_csv(path, encoding=encoding)
            except Exception as exc:  # pragma: no cover - only used for malformed vendor files
                errors.append(exc)
        raise RuntimeError(f"Could not read Tennis-Data CSV {path}: {errors[-1]}")

    @staticmethod
    def _tour(frame: pd.DataFrame, path: Path) -> pd.Series:
        if "Tour" in frame:
            return frame["Tour"].fillna("").astype(str).str.upper()
        filename = path.name.casefold()
        if "wta" in filename:
            return pd.Series("WTA", index=frame.index)
        return pd.Series("ATP", index=frame.index)

    @staticmethod
    def _commence_time(frame: pd.DataFrame) -> pd.Series:
        dates = pd.to_datetime(frame.get("Date"), errors="coerce", dayfirst=True)
        if "Time" in frame:
            combined = pd.to_datetime(
                dates.dt.date.astype(str) + " " + frame["Time"].fillna("12:00").astype(str),
                errors="coerce",
            )
            return combined.fillna(dates.dt.normalize() + pd.Timedelta(hours=12))
        return dates.dt.normalize() + pd.Timedelta(hours=12)

    def load(self) -> tuple[pd.DataFrame, TennisDataLoadReport]:
        files = sorted(self.directory.glob("**/*.csv")) if self.directory.exists() else []
        rows: list[dict] = []
        source_rows = 0
        skipped_files = 0
        for path in files:
            frame = self._read(path)
            required = {"Date", "Winner", "Loser"}
            if not required.issubset(frame.columns):
                skipped_files += 1
                continue
            source_rows += len(frame)
            commence = self._commence_time(frame)
            tours = self._tour(frame, path)
            tournaments = frame.get("Tournament", pd.Series("", index=frame.index)).fillna("")
            for bookmaker, (winner_col, loser_col) in _ODDS_PAIRS.items():
                if winner_col not in frame or loser_col not in frame:
                    continue
                winner_odds = pd.to_numeric(frame[winner_col], errors="coerce")
                loser_odds = pd.to_numeric(frame[loser_col], errors="coerce")
                valid = (
                    commence.notna()
                    & frame["Winner"].notna()
                    & frame["Loser"].notna()
                    & winner_odds.gt(1.0)
                    & loser_odds.gt(1.0)
                )
                for index in frame.index[valid]:
                    match_time = pd.Timestamp(commence.loc[index])
                    rows.append(
                        {
                            "provider_event_id": "",
                            "snapshot_time": match_time - pd.Timedelta(minutes=1),
                            "commence_time": match_time,
                            "tour": str(tours.loc[index]),
                            "tournament": str(tournaments.loc[index]),
                            "player1": str(frame.loc[index, "Winner"]),
                            "player2": str(frame.loc[index, "Loser"]),
                            "bookmaker": bookmaker,
                            "decimal_odds1": float(winner_odds.loc[index]),
                            "decimal_odds2": float(loser_odds.loc[index]),
                            "source": "tennis_data",
                            "is_closing": True,
                            "source_file": str(path),
                        }
                    )
        snapshots = pd.DataFrame(rows)
        if not snapshots.empty:
            snapshots = snapshots.replace([np.inf, -np.inf], np.nan).dropna(
                subset=["snapshot_time", "commence_time", "player1", "player2", "decimal_odds1", "decimal_odds2"]
            )
            snapshots = snapshots.sort_values(
                ["commence_time", "player1", "player2", "bookmaker"]
            ).reset_index(drop=True)
        return snapshots, TennisDataLoadReport(
            files=len(files),
            source_rows=source_rows,
            odds_rows=len(snapshots),
            skipped_files=skipped_files,
        )
