import pandas as pd

from tennis_predictor.ui.view_models import prepare_historical_replay


def test_historical_replay_uses_saved_walk_forward_probabilities():
    predictions = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "date": "2025-07-01",
                "tour": "ATP",
                "tier": "Slam",
                "surface": "Grass",
                "tournament": "Wimbledon",
                "round": "R128",
                "player1": "Player A",
                "player2": "Player B",
                "actual": 0,
                "probability": 0.64,
                "fold": 3,
                "model": "boosted",
                "retirement": False,
                "walkover": False,
            },
            {
                "match_id": "m1",
                "date": "2025-07-01",
                "tour": "ATP",
                "tier": "Slam",
                "surface": "Grass",
                "tournament": "Wimbledon",
                "round": "R128",
                "player1": "Player A",
                "player2": "Player B",
                "actual": 0,
                "probability": 0.57,
                "fold": 3,
                "model": "elo_baseline",
                "retirement": False,
                "walkover": False,
            },
        ]
    )

    replay = prepare_historical_replay(predictions)

    assert len(replay) == 1
    row = replay.iloc[0]
    assert row["boosted_probability"] == 0.64
    assert row["elo_probability"] == 0.57
    assert row["actual"] == 0
    assert row["fold"] == 3


def test_historical_replay_requires_saved_prediction_schema():
    bad = pd.DataFrame({"match_id": ["m1"]})

    try:
        prepare_historical_replay(bad)
    except ValueError as exc:
        assert "walk-forward prediction columns" in str(exc)
    else:
        raise AssertionError("Expected malformed prediction data to be rejected")
