import pandas as pd
import pytest

from tennis_predictor.config import FeatureConfig, FilterConfig
from tennis_predictor.features import FeatureEngineer


def _matches():
    return pd.DataFrame(
        [
            {
                "match_id": "m1", "tour": "ATP", "date": pd.Timestamp("2024-01-01"),
                "event_date": pd.Timestamp("2024-01-01"), "date_is_estimated": False,
                "match_num": 1, "tournament": "Test", "tier": "250", "round": "R32",
                "surface": "Hard", "player1": "A", "player2": "B", "player1_id": "ATP:1",
                "player2_id": "ATP:2", "player1_age": 25.0, "player2_age": 25.0,
                "player1_rank": 10, "player2_rank": 20, "player1_won": 1,
                "retirement": False, "walkover": False, "retiring_player_id": None,
                "dead_rubber": False, "already_qualified": False, "low_tier_before_slam": False,
                "large_line_move": False,
            },
            {
                "match_id": "m2", "tour": "ATP", "date": pd.Timestamp("2024-01-08"),
                "event_date": pd.Timestamp("2024-01-08"), "date_is_estimated": False,
                "match_num": 1, "tournament": "Test2", "tier": "250", "round": "R32",
                "surface": "Hard", "player1": "A", "player2": "B", "player1_id": "ATP:1",
                "player2_id": "ATP:2", "player1_age": 25.0, "player2_age": 25.0,
                "player1_rank": 10, "player2_rank": 20, "player1_won": 0,
                "retirement": False, "walkover": False, "retiring_player_id": None,
                "dead_rubber": False, "already_qualified": False, "low_tier_before_slam": False,
                "large_line_move": False,
            },
        ]
    )


def test_current_match_does_not_update_its_own_features():
    features = FeatureEngineer(FeatureConfig(), FilterConfig()).transform(_matches())
    first = features.iloc[0]
    second = features.iloc[1]
    assert first.elo_diff == pytest.approx(0.0)
    assert first.tour_matches1 == 0
    assert second.elo_diff > 0
    assert second.tour_matches1 == 1
    assert second.h2h_matches == 1
