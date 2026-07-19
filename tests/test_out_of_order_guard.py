import pandas as pd
import pytest

from tennis_predictor.config import AppConfig
from tennis_predictor.data.names import NameResolver, PlayerIdentity
from tennis_predictor.pipeline import TennisPredictionSystem


def test_incremental_update_rejects_unseen_past_match():
    system = TennisPredictionSystem(AppConfig())
    system.fitted = True
    system.name_resolver = NameResolver(
        [
            PlayerIdentity("a", "Player A", "player a", "ATP"),
            PlayerIdentity("b", "Player B", "player b", "ATP"),
        ]
    )
    system.feature_engineer.state.as_of = pd.Timestamp("2026-07-15")
    frame = pd.DataFrame(
        [
            {
                "match_id": "old-unseen",
                "tour": "ATP",
                "date": pd.Timestamp("2026-06-08"),
                "event_date": pd.Timestamp("2026-06-08"),
                "date_is_estimated": False,
                "surface": "Unknown",
                "tier": "ITF",
                "player1": "Player A",
                "player2": "Player B",
                "player1_id": "a",
                "player2_id": "b",
                "player1_won": 1,
                "match_num": 1,
            }
        ]
    )

    with pytest.raises(ValueError, match="Out-of-order"):
        system.update_state(frame)
