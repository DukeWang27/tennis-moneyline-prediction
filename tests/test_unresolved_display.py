import pandas as pd

from tennis_predictor.ui.view_models import clean_unresolved_fixtures


def test_unresolved_display_hides_nested_provider_payloads():
    raw = pd.DataFrame(
        [
            {
                "event_key": "123",
                "event_date": "2026-07-15",
                "event_time": "10:00",
                "event_first_player": "Player A",
                "event_second_player": "Player B",
                "tournament_name": "M25 Example",
                "tour": "ATP",
                "tier": "ITF",
                "event_status": "Finished",
                "unresolved_reason": "unknown surface",
                "event_game_result": [{"game": 1}],
                "event_point_by_point": [{"point": 1}],
            }
        ]
    )

    display = clean_unresolved_fixtures(raw, limit=None)

    assert display.columns.tolist() == [
        "event_date",
        "event_time",
        "matchup",
        "tournament",
        "tour",
        "tier",
        "status",
        "reason",
    ]
    assert display.iloc[0]["matchup"] == "Player A vs Player B"
    assert display.iloc[0]["reason"] == "unknown surface"
    assert not any("object" in str(value).lower() for value in display.iloc[0].tolist())
