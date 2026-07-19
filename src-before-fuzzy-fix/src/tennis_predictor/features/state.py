from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque

import numpy as np
import pandas as pd

from tennis_predictor.config import FeatureConfig, FilterConfig


def _new_h2h_record():
    """Top-level factory so TennisState remains joblib/pickle serializable."""
    return defaultdict(int)


@dataclass(slots=True)
class PlayerState:
    overall_elo: float
    surface_elo: dict[str, float] = field(default_factory=dict)
    surface_matches: dict[str, int] = field(default_factory=dict)
    total_matches: int = 0
    tour_matches: int = 0
    first_date: pd.Timestamp | None = None
    last_date: pd.Timestamp | None = None
    results: Deque[tuple[pd.Timestamp, float, float]] = field(default_factory=lambda: deque(maxlen=100))
    incidents: Deque[pd.Timestamp] = field(default_factory=deque)
    match_dates: Deque[pd.Timestamp] = field(default_factory=lambda: deque(maxlen=30))
    age_last: float | None = None


class TennisState:
    """Mutable pre-match state. Snapshot first, update second."""

    def __init__(self, feature_config: FeatureConfig, filter_config: FilterConfig):
        self.cfg = feature_config
        self.filter_cfg = filter_config
        self.players: dict[str, PlayerState] = {}
        self.h2h: dict[tuple[str, str], dict[str, int]] = defaultdict(_new_h2h_record)
        self.as_of: pd.Timestamp | None = None
        self.processed_match_ids: set[str] = set()

    def player(self, player_id: str) -> PlayerState:
        if player_id not in self.players:
            self.players[player_id] = PlayerState(overall_elo=self.cfg.initial_elo)
        return self.players[player_id]

    def _surface_rating(self, state: PlayerState, surface: str) -> tuple[float, float, int]:
        surface_elo = state.surface_elo.get(surface, self.cfg.initial_elo)
        n = state.surface_matches.get(surface, 0)
        weight = n / (n + self.cfg.surface_prior_matches) if n > 0 else 0.0
        blended = weight * surface_elo + (1.0 - weight) * state.overall_elo
        return blended, surface_elo, n

    def _recent_form(self, state: PlayerState, date: pd.Timestamp, n: int) -> float:
        values = list(state.results)[-n:]
        if not values:
            return 0.5
        numerator = 0.0
        denominator = 0.0
        half_life = max(self.cfg.recent_form_half_life_days, 1e-6)
        for when, result, match_weight in values:
            days = max((date - when).days, 0)
            recency = math.exp(-math.log(2.0) * days / half_life)
            weight = recency * match_weight
            numerator += result * weight
            denominator += weight
        return numerator / denominator if denominator > 0 else 0.5

    @staticmethod
    def _prune_before(queue: Deque[pd.Timestamp], cutoff: pd.Timestamp) -> None:
        while queue and queue[0] < cutoff:
            queue.popleft()

    def snapshot(
        self,
        player1_id: str,
        player2_id: str,
        *,
        date: pd.Timestamp,
        surface: str,
        tier: str,
        player1_age: float | None = None,
        player2_age: float | None = None,
    ) -> dict[str, float | int | bool]:
        date = pd.Timestamp(date)
        p1 = self.player(player1_id)
        p2 = self.player(player2_id)
        r1, surface_raw1, surface_n1 = self._surface_rating(p1, surface)
        r2, surface_raw2, surface_n2 = self._surface_rating(p2, surface)

        incident_cutoff = date - pd.Timedelta(days=365)
        self._prune_before(p1.incidents, incident_cutoff)
        self._prune_before(p2.incidents, incident_cutoff)
        fatigue_cutoff = date - pd.Timedelta(days=self.filter_cfg.fatigue_window_days)
        recent_dates1 = [d for d in p1.match_dates if fatigue_cutoff <= d < date]
        recent_dates2 = [d for d in p2.match_dates if fatigue_cutoff <= d < date]

        key = tuple(sorted((player1_id, player2_id)))
        record = self.h2h[key]
        h2h_p1_wins = record.get(player1_id, 0)
        h2h_p2_wins = record.get(player2_id, 0)
        h2h_n = h2h_p1_wins + h2h_p2_wins
        h2h_diff = 0.0
        if h2h_n >= self.cfg.h2h_min_matches:
            h2h_diff = (h2h_p1_wins - h2h_p2_wins) / h2h_n

        rest1 = self.cfg.rest_days_cap if p1.last_date is None else min(max((date - p1.last_date).days, 0), self.cfg.rest_days_cap)
        rest2 = self.cfg.rest_days_cap if p2.last_date is None else min(max((date - p2.last_date).days, 0), self.cfg.rest_days_cap)
        years1 = 0.0 if p1.first_date is None else max((date - p1.first_date).days / 365.25, 0.0)
        years2 = 0.0 if p2.first_date is None else max((date - p2.first_date).days / 365.25, 0.0)

        if player1_age is not None and pd.notna(player1_age):
            age1 = float(player1_age)
        elif p1.age_last is not None:
            elapsed = 0.0 if p1.last_date is None else max((date - p1.last_date).days / 365.25, 0.0)
            age1 = p1.age_last + elapsed
        else:
            age1 = np.nan
        if player2_age is not None and pd.notna(player2_age):
            age2 = float(player2_age)
        elif p2.age_last is not None:
            elapsed = 0.0 if p2.last_date is None else max((date - p2.last_date).days / 365.25, 0.0)
            age2 = p2.age_last + elapsed
        else:
            age2 = np.nan
        return {
            "elo_overall1": p1.overall_elo,
            "elo_overall2": p2.overall_elo,
            "elo_surface_raw1": surface_raw1,
            "elo_surface_raw2": surface_raw2,
            "surface_matches1": surface_n1,
            "surface_matches2": surface_n2,
            "elo_blended1": r1,
            "elo_blended2": r2,
            "elo_diff": r1 - r2,
            "age1": age1,
            "age2": age2,
            "age_diff": age1 - age2 if np.isfinite(age1) and np.isfinite(age2) else np.nan,
            "tour_matches1": p1.tour_matches,
            "tour_matches2": p2.tour_matches,
            "tour_matches_diff": p1.tour_matches - p2.tour_matches,
            "years_on_tour1": years1,
            "years_on_tour2": years2,
            "years_on_tour_diff": years1 - years2,
            "injury_incidents_12m1": len(p1.incidents),
            "injury_incidents_12m2": len(p2.incidents),
            "injury_incidents_diff": len(p1.incidents) - len(p2.incidents),
            "form10_1": self._recent_form(p1, date, 10),
            "form10_2": self._recent_form(p2, date, 10),
            "form20_1": self._recent_form(p1, date, 20),
            "form20_2": self._recent_form(p2, date, 20),
            "form10_diff": self._recent_form(p1, date, 10) - self._recent_form(p2, date, 10),
            "form20_diff": self._recent_form(p1, date, 20) - self._recent_form(p2, date, 20),
            "h2h_matches": h2h_n,
            "h2h_diff": h2h_diff,
            "rest_days1": rest1,
            "rest_days2": rest2,
            "rest_days_diff": rest1 - rest2,
            "matches_prev_window1": len(recent_dates1),
            "matches_prev_window2": len(recent_dates2),
            "fatigue_player1": len(recent_dates1) >= self.filter_cfg.fatigue_matches,
            "fatigue_player2": len(recent_dates2) >= self.filter_cfg.fatigue_matches,
            "pre_slam_rest_risk_player1": False,
            "pre_slam_rest_risk_player2": False,
            "surface_code": {"Hard": 0, "Clay": 1, "Grass": 2, "Carpet": 3}.get(surface, 4),
            "tier_code": {"Slam": 0, "Masters": 1, "500": 2, "250": 3, "Challenger": 4, "ITF": 5}.get(tier, 6),
        }

    def update(self, row: pd.Series) -> None:
        match_id = str(row.get("match_id", ""))
        if match_id and match_id in getattr(self, "processed_match_ids", set()):
            return
        if not hasattr(self, "processed_match_ids"):
            self.processed_match_ids = set()
        date = pd.Timestamp(row.get("chronology_ts", row["date"]))
        p1_id, p2_id = str(row["player1_id"]), str(row["player2_id"])
        p1, p2 = self.player(p1_id), self.player(p2_id)
        surface = str(row["surface"])
        outcome1 = float(row["player1_won"])
        outcome2 = 1.0 - outcome1
        if bool(row.get("walkover", False)) or bool(row.get("defaulted", False)) or bool(row.get("void", False)):
            match_weight = self.cfg.walkover_elo_weight
        elif bool(row.get("retirement", False)):
            match_weight = self.cfg.retirement_elo_weight
        else:
            match_weight = 1.0

        surface_known = surface in {"Hard", "Clay", "Grass", "Carpet"}
        if surface_known:
            rating1, _, _ = self._surface_rating(p1, surface)
            rating2, _, _ = self._surface_rating(p2, surface)
        else:
            # Unknown-court matches still update general strength and sequential
            # state, but never create or alter an "Unknown" surface rating.
            rating1, rating2 = p1.overall_elo, p2.overall_elo
        expected1 = 1.0 / (1.0 + 10.0 ** ((rating2 - rating1) / 400.0))
        delta = self.cfg.elo_k * match_weight * (outcome1 - expected1)
        p1.overall_elo += delta
        p2.overall_elo -= delta

        if surface_known:
            s1 = p1.surface_elo.get(surface, self.cfg.initial_elo)
            s2 = p2.surface_elo.get(surface, self.cfg.initial_elo)
            expected_surface1 = 1.0 / (1.0 + 10.0 ** ((s2 - s1) / 400.0))
            surface_delta = self.cfg.elo_k * match_weight * (outcome1 - expected_surface1)
            p1.surface_elo[surface] = s1 + surface_delta
            p2.surface_elo[surface] = s2 - surface_delta
        if match_weight > 0:
            if surface_known:
                p1.surface_matches[surface] = p1.surface_matches.get(surface, 0) + 1
                p2.surface_matches[surface] = p2.surface_matches.get(surface, 0) + 1
            p1.results.append((date, outcome1, match_weight))
            p2.results.append((date, outcome2, match_weight))
            p1.match_dates.append(date)
            p2.match_dates.append(date)
            self.h2h[tuple(sorted((p1_id, p2_id)))][p1_id if outcome1 == 1 else p2_id] += 1

        if match_weight > 0:
            for state, pid, age in (
                (p1, p1_id, row.get("player1_age", np.nan)),
                (p2, p2_id, row.get("player2_age", np.nan)),
            ):
                state.total_matches += 1
                if str(row.get("tier", "")) not in {"Challenger", "ITF"}:
                    state.tour_matches += 1
                state.first_date = date if state.first_date is None else min(state.first_date, date)
                state.last_date = date if state.last_date is None else max(state.last_date, date)
                if pd.notna(age):
                    state.age_last = float(age)
                if str(row.get("retiring_player_id", "")) == pid:
                    state.incidents.append(date)
        self.as_of = date if self.as_of is None else max(self.as_of, date)
        if match_id:
            self.processed_match_ids.add(match_id)
