from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from tennis_predictor.config import AppConfig
from tennis_predictor.data.names import NameResolver, PlayerIdentity, normalize_name
from tennis_predictor.features import FeatureEngineer
from tennis_predictor.filters import FilterEngine, GrandSlamCalendar
from tennis_predictor.quality import evaluate_match_sample_quality
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

    def register_player(self, player_id: str, name: str, tour: str) -> PlayerIdentity:
        if self.name_resolver is None:
            raise RuntimeError("Fit or load the system before registering players.")
        existing = self.name_resolver.by_id.get(str(player_id))
        if existing is not None:
            return existing
        identity = PlayerIdentity(
            player_id=str(player_id),
            name=str(name),
            normalized_name=normalize_name(str(name)),
            tour=str(tour).upper(),
        )
        self.name_resolver.register(identity)
        self.feature_engineer.state.player(identity.player_id)
        return identity

    def update_state(self, completed_matches: pd.DataFrame) -> pd.DataFrame:
        """Increment player state with completed matches; do not refit LightGBM.

        Rows are passed through the existing FeatureEngineer in chronological order,
        preserving the snapshot-first/update-second leakage guard. Replayed fixture IDs
        are ignored by TennisState.
        """
        if not self.fitted or self.name_resolver is None:
            raise RuntimeError("Fit or load a TennisPredictionSystem before updating state.")
        if completed_matches.empty:
            return pd.DataFrame()
        frame = completed_matches.copy()
        required = {
            "match_id", "tour", "date", "event_date", "date_is_estimated",
            "surface", "tier", "player1", "player2", "player1_id",
            "player2_id", "player1_won", "match_num",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Incremental matches are missing columns: {sorted(missing)}")
        frame = frame.dropna(subset=["date", "player1_won"]).copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.dropna(subset=["date"]).copy()
        frame["surface"] = frame["surface"].fillna("Unknown").astype(str).str.title()
        if frame.empty:
            return pd.DataFrame()

        processed = getattr(self.feature_engineer.state, "processed_match_ids", set())
        unseen = frame[~frame["match_id"].astype(str).isin(processed)].copy()
        state_date = self.feature_engineer.state.as_of
        if state_date is not None and not unseen.empty:
            earliest = unseen["date"].min().normalize()
            if earliest < pd.Timestamp(state_date).normalize():
                raise ValueError(
                    "Out-of-order completed matches would leak future state backward. "
                    "Rebuild the live model from the frozen pre-live artifact instead."
                )
        for row in frame.itertuples(index=False):
            self.register_player(str(row.player1_id), str(row.player1), str(row.tour))
            self.register_player(str(row.player2_id), str(row.player2), str(row.tour))
        frame = FilterEngine(self.config.filters).add_static_flags(frame)
        features = self.feature_engineer.transform(frame, reset=False)
        if self.calendar is None:
            self.calendar = GrandSlamCalendar.from_matches(frame)
        else:
            new_dates = GrandSlamCalendar.from_matches(frame).dates
            self.calendar.dates = sorted(set(self.calendar.dates).union(new_dates))
        return features


    def score_feature_rows(self, features: pd.DataFrame) -> pd.DataFrame:
        """Score pre-match rows without mutating player state.

        FeatureEngineer creates each snapshot before applying that match result, so
        these probabilities are safe to persist in the sequential replay ledger.
        """
        if features is None or features.empty:
            return pd.DataFrame()
        if not self.fitted or self.boosted_model is None:
            raise RuntimeError("Fit or load the system before scoring feature rows.")
        scored = features.copy()
        age_adjustment = self.age_curve.elo_adjustment(
            scored["age1"], scored["age2"], self.elo_model.scale
        )
        scored["elo_age_adjusted_diff"] = scored["elo_diff"].astype(float) + age_adjustment
        scored["elo_prob"] = self.elo_model.predict_proba(scored["elo_age_adjusted_diff"])
        raw = self.boosted_model.predict_proba(scored)
        scored["p_model1"] = self.calibrator.predict(raw)
        scored["p_elo1"] = scored["elo_prob"]
        return scored

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
        state1 = self.feature_engineer.state.players.get(identity1.player_id)
        state2 = self.feature_engineer.state.players.get(identity2.player_id)
        live_cfg = getattr(self.config, "live_data", None)
        sample_quality = evaluate_match_sample_quality(
            state1,
            state2,
            player1=identity1.name,
            player2=identity2.name,
            surface=surface,
            match_date=date,
            absolute_min_matches=getattr(live_cfg, "cold_start_min_matches", 8),
            recommended_matches=getattr(live_cfg, "recommended_prediction_matches", 20),
            recommended_surface_matches=getattr(live_cfg, "recommended_surface_matches", 5),
            max_staleness_days=getattr(live_cfg, "max_player_staleness_days", 120),
        )
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
        feature_snapshot_keys = [
            "elo_overall1", "elo_overall2", "elo_surface_raw1", "elo_surface_raw2",
            "surface_matches1", "surface_matches2", "elo_blended1", "elo_blended2",
            "elo_diff", "age1", "age2", "age_diff", "tour_matches1", "tour_matches2",
            "years_on_tour1", "years_on_tour2", "injury_incidents_12m1",
            "injury_incidents_12m2", "form10_1", "form10_2", "form20_1", "form20_2",
            "h2h_matches", "h2h_diff", "rest_days1", "rest_days2",
            "matches_prev_window1", "matches_prev_window2",
        ]
        feature_snapshot = {
            key: (None if pd.isna(features.get(key)) else features.get(key))
            for key in feature_snapshot_keys
        }
        feature_snapshot.update(
            {
                "total_matches1": sample_quality.player1.total_matches,
                "total_matches2": sample_quality.player2.total_matches,
                "last_match_date1": sample_quality.player1.last_match_date,
                "last_match_date2": sample_quality.player2.last_match_date,
                "staleness_days1": sample_quality.player1.staleness_days,
                "staleness_days2": sample_quality.player2.staleness_days,
            }
        )
        return {
            "player1": identity1.name,
            "player2": identity2.name,
            "date": date.date().isoformat(),
            "surface": surface,
            "tournament_tier": tier,
            "player1_win_probability": probability,
            "player2_win_probability": 1.0 - probability,
            "elo_probability": features["elo_prob"],
            "feature_snapshot": feature_snapshot,
            "sample_quality": sample_quality.to_dict(),
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
