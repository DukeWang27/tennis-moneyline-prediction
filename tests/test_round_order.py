import pandas as pd

from tennis_predictor.config import FeatureConfig, FilterConfig
from tennis_predictor.features.chronology import enforce_round_order
from tennis_predictor.features.engineer import FeatureEngineer


def _event() -> pd.DataFrame:
    # R32 has an exact (odds) date of Jan 10; the R16 and QF kept Sackmann's estimates, which fall earlier.
    return pd.DataFrame({
        "match_id": ["ATP_2023-1_1", "ATP_2023-1_2", "ATP_2023-1_3", "ATP_2023-2_1"],
        "date": pd.to_datetime(["2023-01-10", "2023-01-09", "2023-01-09", "2023-01-05"]),
        "round": ["R32", "R16", "QF", "R16"],
    })


def test_later_rounds_never_precede_earlier_ones():
    out = enforce_round_order(_event()).set_index("match_id")
    assert out.loc["ATP_2023-1_2", "chronology_ts"] == pd.Timestamp("2023-01-10")
    assert out.loc["ATP_2023-1_3", "chronology_ts"] == pd.Timestamp("2023-01-10")
    assert out.loc["ATP_2023-1_3", "round_order"] > out.loc["ATP_2023-1_2", "round_order"]
    assert out.loc["ATP_2023-2_1", "chronology_ts"] == pd.Timestamp("2023-01-05")  # other events untouched


def test_next_round_cannot_leak_into_pre_match_state():
    # Player "a" wins the R32 (exact date Jan 10) and plays the R16, whose estimated date is Jan 9.
    # Before the fix the R16 was processed first, so a's R32 features already counted the R16.
    matches = pd.DataFrame({
        "match_id": ["ATP_2023-1_9", "ATP_2023-1_1"],
        "date": pd.to_datetime(["2023-01-10", "2023-01-09"]),
        "event_date": pd.to_datetime(["2023-01-07", "2023-01-07"]),
        "round": ["R32", "R16"], "match_num": [9, 1], "tour": "ATP", "tier": "250", "surface": "Hard",
        "tournament": "X", "player1_id": ["a", "a"], "player2_id": ["b", "c"], "player1_won": [1, 0],
        "player1_age": 25.0, "player2_age": 25.0, "player1_rank": 50.0, "player2_rank": 60.0,
        "date_is_estimated": [False, True],
    })
    features = FeatureEngineer(FeatureConfig(), FilterConfig()).transform(matches).set_index("match_id")
    assert features.loc["ATP_2023-1_9", "elo_overall1"] == FeatureConfig().initial_elo  # a's first match: no history
