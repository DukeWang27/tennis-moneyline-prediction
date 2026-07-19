import pandas as pd

from tennis_predictor.data.match_consolidation import (
    CanonicalMatchConsolidator,
)
from tennis_predictor.storage import IdentityStore


def _reverse_source_row(row: pd.Series) -> pd.Series:
    output = row.copy()

    pairs = [
        ("player1", "player2"),
        ("player1_id", "player2_id"),
        ("player1_canonical_id", "player2_canonical_id"),
        ("player1_rank", "player2_rank"),
        ("closing_odds_player1", "closing_odds_player2"),
    ]

    for left, right in pairs:
        left_value = output.get(left)
        right_value = output.get(right)
        output[left] = right_value
        output[right] = left_value

    output["player1_won"] = 1 - int(output["player1_won"])
    return output


def test_orientation_is_invariant_to_source_winner_position():
    original = pd.Series(
        {
            "tour": "ATP",
            "player1": "Winner Name",
            "player2": "Loser Name",
            "player1_id": "canonical:1",
            "player2_id": "canonical:2",
            "player1_canonical_id": 1,
            "player2_canonical_id": 2,
            "player1_rank": 10,
            "player2_rank": 50,
            "closing_odds_player1": 1.40,
            "closing_odds_player2": 3.20,
            "player1_won": 1,
        }
    )

    reversed_source = _reverse_source_row(original)

    oriented_original = CanonicalMatchConsolidator._orient(
        original
    )
    oriented_reversed = CanonicalMatchConsolidator._orient(
        reversed_source
    )

    columns = [
        "player1",
        "player2",
        "player1_id",
        "player2_id",
        "player1_canonical_id",
        "player2_canonical_id",
        "player1_rank",
        "player2_rank",
        "closing_odds_player1",
        "closing_odds_player2",
        "player1_won",
    ]

    assert (
        oriented_original[columns].to_dict()
        == oriented_reversed[columns].to_dict()
    )


def test_date_window_is_applied_before_identity_resolution(
    tmp_path,
):
    store = IdentityStore(tmp_path / "identity.db")
    consolidator = CanonicalMatchConsolidator(store)

    source = pd.DataFrame(
        {
            "date": [
                "2022-12-31",
                "2023-01-15",
                "2024-01-01",
            ],
            "player1": ["Old A", "Current A", "Future A"],
            "player2": ["Old B", "Current B", "Future B"],
            "tour": ["ATP", "ATP", "ATP"],
        }
    )

    captured = {}

    def fake_canonicalize(sources):
        captured["rows"] = sum(
            len(frame)
            for frame, _ in sources
        )
        captured["dates"] = pd.concat(
            [frame["date"] for frame, _ in sources],
            ignore_index=True,
        )
        return pd.DataFrame(), pd.DataFrame()

    consolidator.canonicalize = fake_canonicalize

    _, _, report, _ = consolidator.consolidate(
        [(source, "test")],
        start_date="2023-01-01",
        end_date="2023-12-31",
    )

    assert captured["rows"] == 1
    assert captured["dates"].dt.date.astype(str).tolist() == [
        "2023-01-15"
    ]
    assert report.input_rows == 1
