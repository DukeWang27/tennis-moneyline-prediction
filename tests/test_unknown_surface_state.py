import pandas as pd

from tennis_predictor.config import FeatureConfig, FilterConfig
from tennis_predictor.features.state import TennisState


def test_unknown_surface_updates_general_state_but_not_surface_elo():
    state = TennisState(FeatureConfig(), FilterConfig())
    row = pd.Series(
        {
            "match_id": "m1",
            "date": pd.Timestamp("2026-07-01"),
            "player1_id": "a",
            "player2_id": "b",
            "surface": "Unknown",
            "player1_won": 1,
            "tier": "ITF",
            "walkover": False,
            "retirement": False,
            "player1_age": 22.0,
            "player2_age": 23.0,
            "retiring_player_id": pd.NA,
        }
    )

    state.update(row)

    player_a = state.players["a"]
    player_b = state.players["b"]
    assert player_a.overall_elo > 1500
    assert player_b.overall_elo < 1500
    assert "Unknown" not in player_a.surface_elo
    assert "Unknown" not in player_b.surface_elo
    assert player_a.total_matches == 1
    assert player_b.total_matches == 1
    assert len(player_a.results) == 1
    assert state.h2h[("a", "b")]["a"] == 1
