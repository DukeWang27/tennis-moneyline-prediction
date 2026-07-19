from __future__ import annotations

import pandas as pd

from tennis_predictor.data.itf_women_sync import normalize_sackmann_itf_women


def test_normalize_sackmann_itf_women_filters_tour_qualifying() -> None:
    frame = pd.DataFrame(
        [
            {
                "tourney_id": "2026-W-ITF-GRE-2026-001",
                "tourney_name": "W50 Heraklion",
                "surface": "Clay",
                "tourney_date": 20260309,
                "match_num": 101,
                "winner_id": 1,
                "winner_name": "Winner One",
                "loser_id": 2,
                "loser_name": "Loser Two",
                "round": "R32",
            },
            {
                "tourney_id": "2026-0308",
                "tourney_name": "WTA Qualifying Event",
                "surface": "Hard",
                "tourney_date": 20260309,
                "match_num": 1,
                "winner_id": 3,
                "winner_name": "Tour Winner",
                "loser_id": 4,
                "loser_name": "Tour Loser",
                "round": "Q1",
            },
        ]
    )

    output = normalize_sackmann_itf_women(frame, source_url="https://example.test/2026.csv")

    assert len(output) == 1
    row = output.iloc[0]
    assert row["competition_segment"] == "ITF_WOMEN"
    assert row["surface"] == "Clay"
    assert row["date"] == "2026-03-09"
    assert row["tournament"] == "W50 Heraklion"
    assert row["match_id"] == "2026-W-ITF-GRE-2026-001:101"


def test_normalize_sackmann_itf_women_rejects_unknown_surface() -> None:
    frame = pd.DataFrame(
        [
            {
                "tourney_id": "2026-W-ITF-USA-2026-001",
                "tourney_name": "W15 Example",
                "surface": "",
                "tourney_date": "20260105",
                "match_num": 1,
                "winner_id": 11,
                "winner_name": "A",
                "loser_id": 12,
                "loser_name": "B",
                "round": "R32",
            }
        ]
    )
    output = normalize_sackmann_itf_women(frame, source_url="https://example.test/2026.csv")
    assert output.iloc[0]["surface"] == "Unknown"
