import numpy as np
import pandas as pd

from tennis_predictor.config import AppConfig
from tennis_predictor.data.names import NameResolver
from tennis_predictor.pipeline import TennisPredictionSystem


def _match(match_id: str, when: str, winner: int) -> dict:
    return {
        "match_id": match_id,
        "tour": "ATP",
        "date": pd.Timestamp(when),
        "event_date": pd.Timestamp(when),
        "date_is_estimated": False,
        "tournament": "Test",
        "tournament_level_raw": "ATP Singles",
        "tier": "250",
        "round": "R32",
        "surface": "Hard",
        "match_num": 1 if match_id == "m1" else 2,
        "player1": "Player A",
        "player2": "Player B",
        "player1_id": "ATP:1",
        "player2_id": "ATP:2",
        "player1_age": 25.0,
        "player2_age": 26.0,
        "player1_rank": np.nan,
        "player2_rank": np.nan,
        "winner": "Player A" if winner else "Player B",
        "winner_id": "ATP:1" if winner else "ATP:2",
        "loser_id": "ATP:2" if winner else "ATP:1",
        "player1_won": winner,
        "score": "6-4 6-4",
        "retirement": False,
        "walkover": False,
        "incomplete_match": False,
        "retiring_player_id": pd.NA,
        "best_of": 3,
        "source_file": "test",
    }


def test_incremental_update_uses_existing_feature_engineer_and_is_idempotent():
    config = AppConfig()
    system = TennisPredictionSystem(config)
    base = pd.DataFrame([_match("m1", "2026-06-07", 1)])
    system.feature_engineer.transform(base, reset=True)
    system.name_resolver = NameResolver.from_matches(base)
    system.fitted = True

    before = system.feature_engineer.state.player("ATP:1").overall_elo
    new = pd.DataFrame([_match("api_tennis:2", "2026-06-08", 0)])
    features = system.update_state(new)
    after = system.feature_engineer.state.player("ATP:1").overall_elo
    assert len(features) == 1
    assert after < before
    assert system.feature_engineer.state.as_of == pd.Timestamp("2026-06-08")

    system.update_state(new)
    assert system.feature_engineer.state.player("ATP:1").overall_elo == after
