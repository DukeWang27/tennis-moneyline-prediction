from types import SimpleNamespace

import pandas as pd

from tennis_predictor.quality import evaluate_match_sample_quality


def _state(last_date):
    return SimpleNamespace(
        total_matches=25,
        surface_matches={"Hard": 7},
        last_date=pd.Timestamp(last_date),
    )


def test_sufficient_when_at_least_one_player_is_recent():
    quality = evaluate_match_sample_quality(
        _state("2026-07-10"),
        _state("2025-01-01"),
        player1="Recent Player",
        player2="Stale Player",
        surface="Hard",
        match_date="2026-07-16",
    )
    assert quality.status == "SUFFICIENT"
    assert quality.can_bet is True


def test_limited_when_neither_player_is_recent():
    quality = evaluate_match_sample_quality(
        _state("2025-01-01"),
        _state("2025-02-01"),
        player1="Stale One",
        player2="Stale Two",
        surface="Hard",
        match_date="2026-07-16",
    )
    assert quality.status == "LIMITED"
    assert quality.can_bet is False
