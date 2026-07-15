import pandas as pd
import pytest

from tennis_predictor.config import FeatureConfig, FilterConfig
from tennis_predictor.features import FeatureEngineer


def test_retirement_is_downweighted_and_counted_as_incident():
    match = pd.DataFrame([{
        "match_id": "r1", "tour": "ATP", "date": pd.Timestamp("2024-01-01"),
        "event_date": pd.Timestamp("2024-01-01"), "date_is_estimated": False,
        "match_num": 1, "tournament": "Test", "tier": "250", "round": "R32",
        "surface": "Hard", "player1": "A", "player2": "B", "player1_id": "ATP:1",
        "player2_id": "ATP:2", "player1_age": 25.0, "player2_age": 25.0,
        "player1_rank": 10, "player2_rank": 20, "player1_won": 1,
        "retirement": True, "walkover": False, "retiring_player_id": "ATP:2",
        "dead_rubber": False, "already_qualified": False, "low_tier_before_slam": False,
        "large_line_move": False,
    }])
    config = FeatureConfig(retirement_elo_weight=0.25)
    engineer = FeatureEngineer(config, FilterConfig())
    features = engineer.transform(match)
    assert features.iloc[0].outcome_weight == pytest.approx(0.25)
    assert len(engineer.state.player("ATP:2").incidents) == 1
