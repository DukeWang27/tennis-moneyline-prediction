import pandas as pd

from tennis_predictor.services import prepare_chronological_matches


def test_chronological_preparation_sorts_and_isolates_odds():
    frame = pd.DataFrame(
        {
            "match_id": ["m2", "m1"],
            "match_fingerprint": ["f2", "f1"],
            "date": ["2026-01-02", "2026-01-01"],
            "chronology_ts": ["2026-01-02 10:00", "2026-01-01 10:00"],
            "player1_id": ["canonical:1", "canonical:1"],
            "player2_id": ["canonical:2", "canonical:2"],
            "player1": ["A", "A"],
            "player2": ["B", "B"],
            "player1_won": [0, 1],
            "tour": ["ATP", "ATP"],
            "tier": ["ITF", "ITF"],
            "surface": ["Unknown", "Hard"],
            "closing_odds_player1": [2.0, 1.5],
            "closing_odds_player2": [1.8, 2.8],
        }
    )
    output = prepare_chronological_matches(frame)
    assert output["match_id"].tolist() == ["m1", "m2"]
    assert output["chronology_ts"].is_monotonic_increasing
    assert output.loc[0, "odds_benchmark_type"] == "historical_closing"
