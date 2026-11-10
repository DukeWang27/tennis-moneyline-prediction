from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tennis_predictor.config import BacktestConfig, ModelConfig
from tennis_predictor.features.engineer import FeatureEngineer
from tennis_predictor.models import (
    BoostedProbabilityModel,
    EloLogisticModel,
    EmpiricalAgeCurve,
    ProbabilityCalibrator,
)
from .metrics import evaluate_predictions


@dataclass(slots=True)
class FoldArtifacts:
    fold: int
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    elo_scale: float
    boosted_backend: str


class WalkForwardBacktester:
    def __init__(self, model_config: ModelConfig, backtest_config: BacktestConfig):
        self.model_config = model_config
        self.backtest_config = backtest_config
        self.fold_artifacts: list[FoldArtifacts] = []

    def _folds(self, frame: pd.DataFrame):
        start = frame["date"].min().normalize() + pd.DateOffset(years=self.backtest_config.min_train_years)
        end = frame["date"].max().normalize() + pd.Timedelta(days=1)
        fold = 0
        while start < end:
            test_end = min(start + pd.DateOffset(months=self.backtest_config.test_months), end)
            train_idx = frame["date"] < start
            test_idx = (frame["date"] >= start) & (frame["date"] < test_end)
            if train_idx.sum() and test_idx.sum():
                yield fold, start, test_end, train_idx, test_idx
                fold += 1
            start = test_end

    # Walk-forward data must not be oriented by the winner. With player 1 drawn at random, their win rate is ~50%; a run
    # on 2026-07-18 had 76% (82.8% accuracy, +137% ROI) because the source listed the winner first. That is a leak.
    MAX_PLAYER1_WIN_RATE = 0.65
    MIN_ROWS_FOR_ORIENTATION_CHECK = 500

    def run(self, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        frame = features.sort_values(["date", "match_id"]).reset_index(drop=True).copy()
        if len(frame) >= self.MIN_ROWS_FOR_ORIENTATION_CHECK:
            rate = float(frame["player1_won"].mean())
            if not (1 - self.MAX_PLAYER1_WIN_RATE) <= rate <= self.MAX_PLAYER1_WIN_RATE:
                raise ValueError(
                    f"player1 won {rate:.1%} of {len(frame)} matches. The data looks oriented by the winner, which leaks "
                    "the result into every feature. Randomise which player is player1 before building features."
                )
        feature_columns = FeatureEngineer.numeric_feature_columns()
        predictions: list[pd.DataFrame] = []

        for fold, test_start, test_end, train_idx, test_idx in self._folds(frame):
            historical = frame.loc[train_idx].copy()
            test = frame.loc[test_idx].copy()
            calibration_start = test_start - pd.Timedelta(days=self.backtest_config.calibration_days)
            calibration = historical[historical["date"] >= calibration_start].copy()
            base_train = historical[historical["date"] < calibration_start].copy()
            if len(calibration) < self.backtest_config.min_calibration_rows:
                split = max(int(len(historical) * 0.85), 1)
                base_train = historical.iloc[:split].copy()
                calibration = historical.iloc[split:].copy()
            if len(base_train) < 1000 or len(calibration) < 100:
                continue

            first_elo = EloLogisticModel().fit(
                base_train["elo_diff"],
                base_train["player1_won"],
                base_train["outcome_weight"],
            )
            age_curve = EmpiricalAgeCurve(random_state=self.model_config.random_state).fit(
                base_train["age1"],
                base_train["age2"],
                base_train["elo_diff"],
                base_train["player1_won"],
                base_train["outcome_weight"],
            )
            for part in (base_train, calibration, test):
                part["age_elo_adjustment"] = age_curve.elo_adjustment(
                    part["age1"], part["age2"], first_elo.scale
                )
                part["elo_age_adjusted_diff"] = part["elo_diff"] + part["age_elo_adjustment"]

            elo = EloLogisticModel().fit(
                base_train["elo_age_adjusted_diff"],
                base_train["player1_won"],
                base_train["outcome_weight"],
            )
            for part in (base_train, calibration, test):
                part["elo_prob"] = elo.predict_proba(part["elo_age_adjusted_diff"])

            baseline = self._prediction_frame(test, test["elo_prob"].to_numpy(), fold, "elo_baseline")
            predictions.append(self._add_bets(baseline))

            boosted = BoostedProbabilityModel(
                self.model_config.lightgbm_params,
                random_state=self.model_config.random_state,
            ).fit(
                base_train,
                base_train["player1_won"],
                feature_columns,
                sample_weight=base_train["outcome_weight"],
            )
            raw_cal = boosted.predict_proba(calibration)
            calibrator = ProbabilityCalibrator(self.model_config.calibration).fit(
                raw_cal,
                calibration["player1_won"],
                calibration["outcome_weight"],
            )
            boosted_probability = calibrator.predict(boosted.predict_proba(test))
            boosted_frame = self._prediction_frame(test, boosted_probability, fold, "boosted")
            predictions.append(self._add_bets(boosted_frame))
            self.fold_artifacts.append(
                FoldArtifacts(
                    fold=fold,
                    train_end=historical["date"].max(),
                    test_start=test_start,
                    test_end=test_end,
                    elo_scale=elo.scale,
                    boosted_backend=boosted.backend,
                )
            )

        if not predictions:
            raise RuntimeError("No valid walk-forward folds were produced. Add data or reduce minimum training requirements.")
        result = pd.concat(predictions, ignore_index=True)
        metrics = evaluate_predictions(
            result,
            bootstrap_samples=self.backtest_config.bootstrap_samples,
            random_state=self.backtest_config.random_state,
        )
        return result, metrics

    @staticmethod
    def _prediction_frame(test: pd.DataFrame, probability: np.ndarray, fold: int, model: str) -> pd.DataFrame:
        keep = [
            "match_id", "match_fingerprint", "chronology_ts", "date", "tour", "tier", "competition_segment", "surface",
            "tournament", "round", "is_qualifying", "player1", "player2",
            "player1_won", "retirement", "walkover", "defaulted",
            "eligible_with_filters", "any_filter_flag", "dead_rubber", "already_qualified",
            "fatigue_player1", "fatigue_player2", "pre_slam_rest_risk_player1",
            "pre_slam_rest_risk_player2", "large_line_move", "market_p1_open",
            "market_p1_close", "market_decimal1_open", "market_decimal2_open",
            "market_decimal1_close", "market_decimal2_close", "closing_odds_source",
            "closing_odds_bookmaker", "odds_benchmark_type",
        ]
        output = test.reindex(columns=keep).copy()
        output["actual"] = test["player1_won"].to_numpy()
        output["probability"] = np.asarray(probability)
        output["fold"] = fold
        output["model"] = model
        return output

    def _add_bets(self, frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        output["bet_side"] = 0
        output["flat_stake"] = 0.0
        output["flat_profit"] = 0.0
        output["kelly_stake"] = 0.0
        output["kelly_profit"] = 0.0
        output["clv_probability"] = np.nan
        output["clv_price"] = np.nan
        has_market = output["market_p1_open"].notna()
        edge1 = output["probability"] - output["market_p1_open"]
        edge2 = (1.0 - output["probability"]) - (1.0 - output["market_p1_open"])
        # Retirements, walkovers and defaults are voided (stake returned) on Polymarket and by most books, so they are
        # neither a win nor a loss here. They used to settle as ordinary results.
        flags = output.reindex(columns=["retirement", "walkover", "defaulted"], fill_value=False)
        void = (flags.astype(float).fillna(0.0) > 0).any(axis=1)
        wanted = has_market & ((edge1 >= self.backtest_config.edge_threshold) | (edge2 >= self.backtest_config.edge_threshold))
        output["voided"] = wanted & void
        bet1 = has_market & ~void & (edge1 >= self.backtest_config.edge_threshold)
        bet2 = has_market & ~void & (edge2 >= self.backtest_config.edge_threshold)
        output.loc[bet1, "bet_side"] = 1
        output.loc[bet2, "bet_side"] = 2
        placed = bet1 | bet2
        output.loc[placed, "flat_stake"] = 1.0

        odds = np.where(bet1, output["market_decimal1_open"], output["market_decimal2_open"])
        win = np.where(bet1, output["actual"].eq(1), output["actual"].eq(0))
        output.loc[placed, "flat_profit"] = np.where(win[placed], odds[placed] - 1.0, -1.0)

        p = np.where(bet1, output["probability"], 1.0 - output["probability"])
        b = odds - 1.0
        full_kelly = np.divide(b * p - (1.0 - p), b, out=np.zeros_like(p, dtype=float), where=b > 0)
        stake = np.clip(
            full_kelly * self.backtest_config.kelly_fraction,
            0.0,
            self.backtest_config.max_kelly_fraction,
        )
        output.loc[placed, "kelly_stake"] = stake[placed]
        output.loc[placed, "kelly_profit"] = np.where(
            win[placed], stake[placed] * (odds[placed] - 1.0), -stake[placed]
        )

        close_prob = np.where(bet1, output["market_p1_close"], 1.0 - output["market_p1_close"])
        close_odds = np.where(bet1, output["market_decimal1_close"], output["market_decimal2_close"])
        output.loc[placed, "clv_probability"] = p[placed] - close_prob[placed]
        output.loc[placed, "clv_price"] = odds[placed] / close_odds[placed] - 1.0
        return output
