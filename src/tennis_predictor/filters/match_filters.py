from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from tennis_predictor.config import FilterConfig


@dataclass(slots=True)
class GrandSlamCalendar:
    dates: list[pd.Timestamp]

    @classmethod
    def from_matches(cls, matches: pd.DataFrame) -> "GrandSlamCalendar":
        dates = (
            matches.loc[matches["tier"] == "Slam", "event_date"]
            .dropna()
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
        return cls([pd.Timestamp(x).normalize() for x in dates])

    def days_until_next(self, date: pd.Timestamp) -> int | None:
        day = pd.Timestamp(date).normalize()
        future = [(slam - day).days for slam in self.dates if slam >= day]
        return min(future) if future else None


class FilterEngine:
    """Attach flags without deleting rows; unknown context stays explicit."""

    FLAG_COLUMNS = [
        "dead_rubber",
        "already_qualified",
        "fatigue_player1",
        "fatigue_player2",
        "pre_slam_rest_risk_player1",
        "pre_slam_rest_risk_player2",
        "large_line_move",
    ]

    def __init__(self, config: FilterConfig):
        self.config = config
        self.context_dir = Path(config.context_dir)

    def add_static_flags(self, matches: pd.DataFrame) -> pd.DataFrame:
        frame = matches.copy()
        frame["round_robin_match"] = frame["round"].eq("RR")
        frame = self._merge_context(
            frame,
            self.context_dir / "round_robin_context.csv",
            flag="dead_rubber",
            known="dead_rubber_known",
        )
        frame = self._merge_context(
            frame,
            self.context_dir / "qualification_context.csv",
            flag="already_qualified",
            known="already_qualified_known",
        )
        frame["qualifying_match"] = frame["round"].astype(str).str.startswith("Q")

        calendar = GrandSlamCalendar.from_matches(frame)
        days = frame["date"].map(calendar.days_until_next)
        frame["days_until_next_slam"] = pd.to_numeric(days, errors="coerce")
        frame["low_tier_before_slam"] = (
            frame["tier"].isin(self.config.low_tier_before_slam)
            & frame["days_until_next_slam"].between(1, self.config.pre_slam_window_days)
        )
        return frame

    @staticmethod
    def _merge_context(
        frame: pd.DataFrame,
        path: Path,
        *,
        flag: str,
        known: str,
    ) -> pd.DataFrame:
        frame[flag] = False
        frame[known] = False
        if not path.exists():
            return frame
        context = pd.read_csv(path)
        if not {"match_id", flag}.issubset(context.columns):
            raise ValueError(f"{path} must contain match_id,{flag}")
        mapping = context.set_index("match_id")[flag].astype(bool)
        mask = frame["match_id"].isin(mapping.index)
        frame.loc[mask, flag] = frame.loc[mask, "match_id"].map(mapping)
        frame.loc[mask, known] = True
        return frame

    @classmethod
    def add_eligibility_views(cls, frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        for col in cls.FLAG_COLUMNS:
            if col not in output:
                output[col] = False
        output["any_filter_flag"] = output[cls.FLAG_COLUMNS].fillna(False).any(axis=1)
        output["eligible_with_filters"] = ~output["any_filter_flag"]
        return output
