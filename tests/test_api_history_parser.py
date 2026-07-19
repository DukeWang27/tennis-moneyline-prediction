from tennis_predictor.data.historical_backfill import ApiTennisHistoryParser
from tennis_predictor.storage.identity_store import IdentityStore


def _event(**overrides):
    item = {
        "event_key": "1001",
        "event_date": "2026-07-10",
        "event_time": "14:30",
        "event_first_player": "A. Player",
        "first_player_key": "101",
        "event_second_player": "B. Player",
        "second_player_key": "202",
        "event_winner": "First Player",
        "event_status": "Finished",
        "event_type_type": "Itf Women Singles",
        "tournament_name": "ITF W15 Example Women",
        "tournament_key": "777",
        "tournament_round": "Qualification 1",
        "event_qualification": "True",
        "scores": [{"score_first": "6", "score_second": "4"}],
    }
    item.update(overrides)
    return item


def test_unknown_surface_is_retained_and_id_first(tmp_path):
    parser = ApiTennisHistoryParser(IdentityStore(tmp_path / "identity.db"))
    row, reason = parser.parse_item(_event())
    assert reason is None
    assert row["surface"] == "Unknown"
    assert row["surface_known"] is False
    assert row["competition_segment"] == "ITF_WOMEN"
    assert row["is_qualifying"] is True
    assert row["player1_id"].startswith("api_tennis:")
    assert row["player2_id"].startswith("api_tennis:")


def test_retirement_walkover_and_default_are_flagged(tmp_path):
    parser = ApiTennisHistoryParser(IdentityStore(tmp_path / "identity.db"))
    retired, _ = parser.parse_item(_event(event_status="Retired"))
    walkover, _ = parser.parse_item(_event(event_key="1002", event_status="Walkover"))
    defaulted, _ = parser.parse_item(_event(event_key="1003", event_status="Defaulted"))
    assert retired["retirement"] is True
    assert retired["game_form_eligible"] is False
    assert walkover["walkover"] is True
    assert walkover["game_form_eligible"] is False
    assert defaulted["defaulted"] is True
    assert defaulted["game_form_eligible"] is False
