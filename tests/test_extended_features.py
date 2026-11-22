import numpy as np
import pandas as pd

from tennis_predictor.features.extended import (
    ExtendedFeatureBuilder,
    infer_tournament_countries,
)


def _details(last_match_aces: int) -> pd.DataFrame:
    rows = []
    for i, (w, l, aces) in enumerate([("a", "b", 5), ("b", "c", 3), ("a", "c", 10), ("a", "b", last_match_aces)]):
        rows.append({
            "match_id": f"m{i}", "tourney_id": f"t{i}", "winner_id": w, "loser_id": l, "winner_ht": 190, "loser_ht": 180,
            "winner_hand": "L" if w == "a" else "R", "loser_hand": "R", "winner_ioc": "USA", "loser_ioc": "ESP",
            "winner_entry": "", "loser_entry": "Q", "winner_seed": 1, "loser_seed": np.nan,
            "winner_rank_points": 1000, "loser_rank_points": 500, "minutes": 90,
            "w_ace": aces, "w_df": 2, "w_svpt": 60, "w_1stWon": 30, "w_2ndWon": 12, "w_bpSaved": 2, "w_bpFaced": 3,
            "l_ace": 1, "l_df": 4, "l_svpt": 62, "l_1stWon": 25, "l_2ndWon": 10, "l_bpSaved": 3, "l_bpFaced": 7,
        })
    return pd.DataFrame(rows).set_index("match_id")


def _matches() -> pd.DataFrame:
    return pd.DataFrame({
        "match_id": ["m0", "m1", "m2", "m3"],
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
        "tour": "ATP",
        "player1_id": ["a", "c", "a", "b"],
        "player2_id": ["b", "b", "c", "a"],
        "player1_won": [1, 0, 1, 0],
    })


def test_a_match_does_not_see_its_own_stats():
    low = ExtendedFeatureBuilder(_details(0), {}).build(_matches()).set_index("match_id")
    high = ExtendedFeatureBuilder(_details(40), {}).build(_matches()).set_index("match_id")
    pd.testing.assert_frame_equal(low, high)  # the last match's aces only affect later matches; there are none


def test_history_moves_features_and_sides_are_oriented():
    ext = ExtendedFeatureBuilder(_details(5), {}).build(_matches()).set_index("match_id")
    assert ext.loc["m0", "x_ace_rate1"] == ext.loc["m0", "x_ace_rate2"]  # no history yet: both at the tour prior
    assert ext.loc["m2", "x_ace_rate_diff"] > 0  # "a" served more aces before m2 than "c"
    assert ext.loc["m3", "x_qualifier1"] == 1.0 and ext.loc["m3", "x_qualifier2"] == 0.0  # player1 "b" lost: loser entry Q
    assert ext.loc["m3", "x_lefty2"] == 1.0 and ext.loc["m3", "x_lefty_vs_righty"] == 1.0


def test_tournament_country_from_wildcards():
    details = pd.DataFrame({
        "tourney_id": ["t1", "t1", "t1", "t2"],
        "winner_id": ["p1", "p2", "p3", "p4"], "loser_id": ["q1", "q2", "q3", "q4"],
        "winner_entry": ["WC", "WC", "", "WC"], "loser_entry": ["", "WC", "", ""],
        "winner_ioc": ["FRA", "FRA", "USA", "GBR"], "loser_ioc": ["ESP", "FRA", "ESP", "ESP"],
    })
    assert infer_tournament_countries(details) == {"t1": "FRA"}  # t2 has a single wildcard: left unknown
