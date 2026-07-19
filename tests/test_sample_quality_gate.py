import pandas as pd

from tennis_predictor.features.state import PlayerState
from tennis_predictor.quality import evaluate_match_sample_quality


def _state(total: int, surface: int, last: str) -> PlayerState:
    state = PlayerState(overall_elo=1500.0)
    state.total_matches = total
    state.surface_matches["Hard"] = surface
    state.last_date = pd.Timestamp(last)
    return state


def test_sample_quality_blocks_tiny_itf_history():
    quality = evaluate_match_sample_quality(
        _state(7, 3, "2026-07-10"),
        _state(25, 10, "2026-07-12"),
        player1="Player A",
        player2="Player B",
        surface="Hard",
        match_date="2026-07-16",
    )
    assert quality.status == "INSUFFICIENT"
    assert quality.can_predict is False
    assert quality.can_bet is False


def test_sample_quality_allows_prediction_but_blocks_bet_when_surface_is_sparse():
    quality = evaluate_match_sample_quality(
        _state(30, 2, "2026-07-10"),
        _state(25, 8, "2026-07-12"),
        player1="Player A",
        player2="Player B",
        surface="Hard",
        match_date="2026-07-16",
    )
    assert quality.status == "LIMITED"
    assert quality.can_predict is True
    assert quality.can_bet is False


def test_sample_quality_marks_well_covered_match_sufficient():
    quality = evaluate_match_sample_quality(
        _state(30, 8, "2026-07-10"),
        _state(25, 6, "2026-07-12"),
        player1="Player A",
        player2="Player B",
        surface="Hard",
        match_date="2026-07-16",
    )
    assert quality.status == "SUFFICIENT"
    assert quality.can_predict is True
    assert quality.can_bet is True
