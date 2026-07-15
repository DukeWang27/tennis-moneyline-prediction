from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from tennis_predictor.config import AppConfig
from tennis_predictor.data.names import NameResolver
from tennis_predictor.features import FeatureEngineer
from tennis_predictor.filters import FilterEngine, GrandSlamCalendar
from tennis_predictor.models import (
    BoostedProbabilityModel,
    EloLogisticModel,
    EmpiricalAgeCurve,
    ProbabilityCalibrator,
)


class TennisPredictionSystem:
    """Final fitted state plus probability models for future as-of predictions."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.feature_engineer = FeatureEngineer(config.features, config.filters)
        self.name_resolver: NameResolver | None = None
        self.calendar: GrandSlamCalendar | None = None
        self.elo_model = EloLogisticModel()
        self.age_curve = EmpiricalAgeCurve(random_state=config.model.random_state)
        self.boosted_model: BoostedProbabilityModel | None = None
        self.calibrator = ProbabilityCalibrator(config.model.calibration)
        self.feature_columns = FeatureEngineer.numeric_feature_columns()
        self.fitted = False

    def fit(self, matches: pd.DataFrame, cutoff: str | pd.Timestamp | None = None) -> "TennisPredictionSystem":
        frame = matches.copy()
        if cutoff is not None:
            frame = frame[frame["date"] <= pd.Timestamp(cutoff)].copy()
        frame = FilterEngine(self.config.filters).add_static_flags(frame)
        features = self.feature_engineer.transform(frame, reset=True)
        self.name_resolver = NameResolver.from_matches(frame)
        self.calendar = GrandSlamCalendar.from_matches(frame)

        calibration_start = features["date"].max() - pd.Timedelta(days=self.config.backtest.calibration_days)
        train = features[features["date"] < calibration_start].copy()
        calibration = features[features["date"] >= calibration_start].copy()
        if len(train) < 1000 or len(calibration) < 100:
            split = max(int(len(features) * 0.85), 1)
            train, calibration = features.iloc[:split].copy(), features.iloc[split:].copy()
        if len(train) < 100:
            raise ValueError("Not enough matches to fit the system.")

        initial_elo = EloLogisticModel().fit(
            train["elo_diff"], train["player1_won"], train["outcome_weight"]
        )
        self.age_curve.fit(
            train["age1"],
            train["age2"],
            train["elo_diff"],
            train["player1_won"],
            train["outcome_weight"],
        )
        for part in (train, calibration):
            part["elo_age_adjusted_diff"] = part["elo_diff"] + self.age_curve.elo_adjustment(
                part["age1"], part["age2"], initial_elo.scale
            )
        self.elo_model.fit(
            train["elo_age_adjusted_diff"], train["player1_won"], train["outcome_weight"]
        )
        for part in (train, calibration):
            part["elo_prob"] = self.elo_model.predict_proba(part["elo_age_adjusted_diff"])

        self.boosted_model = BoostedProbabilityModel(
            self.config.model.lightgbm_params,
            random_state=self.config.model.random_state,
        ).fit(
            train,
            train["player1_won"],
            self.feature_columns,
            sample_weight=train["outcome_weight"],
        )
        self.calibrator.fit(
            self.boosted_model.predict_proba(calibration),
            calibration["player1_won"],
            calibration["outcome_weight"],
        )
        self.fitted = True
        return self

    def predict_match(
        self,
        player1: str,
        player2: str,
        surface: str,
        tournament_tier: str,
        match_date: str | pd.Timestamp,
        *,
        tour: str | None = None,
        dead_rubber: bool | None = None,
        already_qualified: bool | None = None,
        large_line_move: bool = False,
    ) -> dict[str, Any]:
        if not self.fitted or self.name_resolver is None or self.boosted_model is None:
            raise RuntimeError("Fit or load a TennisPredictionSystem before predicting.")
        date = pd.Timestamp(match_date)
        state_date = self.feature_engineer.state.as_of
        if state_date is not None and date < state_date.normalize():
            raise ValueError(
                f"Prediction date {date.date()} precedes fitted state {state_date.date()}; "
                "refit with a historical cutoff to avoid leakage."
            )
        identity1 = self.name_resolver.resolve(player1, tour=tour)
        identity2 = self.name_resolver.resolve(player2, tour=tour)
        surface = str(surface).title()
        tier = str(tournament_tier)
        features = self.feature_engineer.state.snapshot(
            identity1.player_id,
            identity2.player_id,
            date=date,
            surface=surface,
            tier=tier,
        )
        days_to_slam = self.calendar.days_until_next(date) if self.calendar else None
        pre_slam = (
            tier in self.config.filters.low_tier_before_slam
            and days_to_slam is not None
            and 1 <= days_to_slam <= self.config.filters.pre_slam_window_days
        )
        features.update(
            {
                "pre_slam_rest_risk_player1": pre_slam,
                "pre_slam_rest_risk_player2": pre_slam,
                "dead_rubber": bool(dead_rubber) if dead_rubber is not None else False,
                "already_qualified": bool(already_qualified) if already_qualified is not None else False,
                "large_line_move": bool(large_line_move),
                "tour_code": {"ATP": 0, "WTA": 1}.get((tour or identity1.tour or "").upper(), 2),
                "rank_diff": np.nan,
                "rank_missing1": 1,
                "rank_missing2": 1,
                "date_estimated_numeric": 0,
            }
        )
        age_adjustment = self.age_curve.elo_adjustment(
            [features["age1"]], [features["age2"]], self.elo_model.scale
        )[0]
        features["elo_age_adjusted_diff"] = features["elo_diff"] + age_adjustment
        features["elo_prob"] = float(
            self.elo_model.predict_proba([features["elo_age_adjusted_diff"]])[0]
        )
        row = pd.DataFrame([features])
        raw = self.boosted_model.predict_proba(row)[0]
        probability = float(self.calibrator.predict([raw])[0])
        flags = {
            "dead_rubber": bool(features["dead_rubber"]),
            "already_qualified": bool(features["already_qualified"]),
            "fatigue_player1": bool(features["fatigue_player1"]),
            "fatigue_player2": bool(features["fatigue_player2"]),
            "pre_slam_rest_risk_player1": bool(pre_slam),
            "pre_slam_rest_risk_player2": bool(pre_slam),
            "large_line_move": bool(large_line_move),
        }
        return {
            "player1": identity1.name,
            "player2": identity2.name,
            "date": date.date().isoformat(),
            "surface": surface,
            "tournament_tier": tier,
            "player1_win_probability": probability,
            "player2_win_probability": 1.0 - probability,
            "elo_probability": features["elo_prob"],
            "active_filter_flags": [name for name, active in flags.items() if active],
            "filter_flags": flags,
            "unknown_context": {
                "dead_rubber": dead_rubber is None,
                "already_qualified": already_qualified is None,
            },
            "as_of": state_date.date().isoformat() if state_date is not None else None,
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "TennisPredictionSystem":
        system = joblib.load(path)
        if not isinstance(system, cls):
            raise TypeError(f"{path} did not contain a TennisPredictionSystem")
        return system
