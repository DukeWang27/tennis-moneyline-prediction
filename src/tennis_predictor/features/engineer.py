from __future__ import annotations

import numpy as np
import pandas as pd

from tennis_predictor.config import FeatureConfig, FilterConfig
from tennis_predictor.filters.match_filters import FilterEngine
from .state import TennisState


class FeatureEngineer:
    """Create one pre-match row at a time, then update state."""

    def __init__(self, feature_config: FeatureConfig, filter_config: FilterConfig):
        self.feature_config = feature_config
        self.filter_config = filter_config
        self.state = TennisState(feature_config, filter_config)

    def transform(self, matches: pd.DataFrame, *, reset: bool = True) -> pd.DataFrame:
        if reset:
            self.state = TennisState(self.feature_config, self.filter_config)
        ordered = matches.sort_values(["date", "event_date", "match_num", "match_id"]).copy()
        rows: list[dict] = []
        for _, match in ordered.iterrows():
            features = self.state.snapshot(
                str(match["player1_id"]),
                str(match["player2_id"]),
                date=pd.Timestamp(match["date"]),
                surface=str(match["surface"]),
                tier=str(match["tier"]),
                player1_age=match.get("player1_age"),
                player2_age=match.get("player2_age"),
            )
            # This is a match-level motivation flag. Both players are marked so
            # downstream testing can remove the match without claiming which one rests.
            pre_slam = bool(match.get("low_tier_before_slam", False))
            features["pre_slam_rest_risk_player1"] = pre_slam
            features["pre_slam_rest_risk_player2"] = pre_slam
            record = match.to_dict()
            record.update(features)
            if bool(match.get("walkover", False)):
                record["outcome_weight"] = self.feature_config.walkover_elo_weight
            elif bool(match.get("retirement", False)):
                record["outcome_weight"] = self.feature_config.retirement_elo_weight
            else:
                record["outcome_weight"] = 1.0
            rows.append(record)
            self.state.update(match)
        output = pd.DataFrame(rows)
        output["tour_code"] = output["tour"].map({"ATP": 0, "WTA": 1}).fillna(2).astype(int)
        output["rank_diff"] = pd.to_numeric(output.get("player2_rank"), errors="coerce") - pd.to_numeric(output.get("player1_rank"), errors="coerce")
        output["rank_missing1"] = output.get("player1_rank").isna().astype(int)
        output["rank_missing2"] = output.get("player2_rank").isna().astype(int)
        output["date_estimated_numeric"] = output["date_is_estimated"].astype(int)
        output = FilterEngine.add_eligibility_views(output)
        return output.sort_values(["date", "match_id"]).reset_index(drop=True)

    @staticmethod
    def numeric_feature_columns() -> list[str]:
        return [
            "elo_diff",
            "elo_prob",
            "elo_age_adjusted_diff",
            "age1",
            "age2",
            "age_diff",
            "tour_matches1",
            "tour_matches2",
            "tour_matches_diff",
            "years_on_tour1",
            "years_on_tour2",
            "years_on_tour_diff",
            "injury_incidents_12m1",
            "injury_incidents_12m2",
            "injury_incidents_diff",
            "form10_1",
            "form10_2",
            "form20_1",
            "form20_2",
            "form10_diff",
            "form20_diff",
            "h2h_matches",
            "h2h_diff",
            "rest_days1",
            "rest_days2",
            "rest_days_diff",
            "matches_prev_window1",
            "matches_prev_window2",
            "surface_matches1",
            "surface_matches2",
            "rank_diff",
            "rank_missing1",
            "rank_missing2",
            "surface_code",
            "tier_code",
            "tour_code",
            "date_estimated_numeric",
            "fatigue_player1",
            "fatigue_player2",
            "pre_slam_rest_risk_player1",
            "pre_slam_rest_risk_player2",
            "dead_rubber",
            "already_qualified",
            "large_line_move",
        ]
