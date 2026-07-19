from types import SimpleNamespace

import pandas as pd

from tennis_predictor.quality import evaluate_match_sample_quality


def _state(last_date):
    return SimpleNamespace(
        total_matches=30,
        surface_matches={"Hard": 10},
        last_date=last_date,
    )


def test_sample_quality_handles_naive_state_and_aware_market_time():
    quality = evaluate_match_sample_quality(
        _state(pd.Timestamp("2026-07-10")),
        _state(pd.Timestamp("2026-07-11")),
        player1="Player A",
        player2="Player B",
        surface="Hard",
        match_date=pd.Timestamp("2026-07-16 20:30:00", tz="America/New_York"),
    )

    assert quality.status == "SUFFICIENT"
    assert quality.player1.staleness_days == 6
    assert quality.player2.staleness_days == 5


def test_sample_quality_handles_aware_state_and_naive_match_date():
    quality = evaluate_match_sample_quality(
        _state(pd.Timestamp("2026-07-10 22:00:00", tz="UTC")),
        _state(pd.Timestamp("2026-07-11 01:00:00", tz="Europe/London")),
        player1="Player A",
        player2="Player B",
        surface="Hard",
        match_date="2026-07-16",
    )

    assert quality.status == "SUFFICIENT"
    assert quality.player1.staleness_days == 6
    assert quality.player2.staleness_days == 5
