from types import SimpleNamespace

import pandas as pd

from tennis_predictor.quality import evaluate_match_sample_quality
from tennis_predictor.services.live_data import _prepare_moneyline_markets
from tennis_predictor.ui import pages


def _state(last_date):
    return SimpleNamespace(
        total_matches=30,
        surface_matches={"Hard": 10},
        last_date=last_date,
    )


def test_timezone_safe_sample_quality():
    result = evaluate_match_sample_quality(
        _state(pd.Timestamp("2026-07-10")),
        _state(pd.Timestamp("2026-07-11", tz="UTC")),
        player1="A",
        player2="B",
        surface="Hard",
        match_date=pd.Timestamp("2026-07-19 12:00", tz="America/New_York"),
    )
    assert result.player1.staleness_days == 9
    assert result.player2.staleness_days == 8


def test_doubles_remain_visible_but_are_not_model_supported():
    markets = pd.DataFrame(
        [
            {
                "market_id": "doubles-1",
                "provider_event_id": "event-1",
                "question": "A/B vs C/D",
                "display_title": "A/B vs C/D",
                "player1": "A/B",
                "player2": "C/D",
                "parse_status": "PARSED",
                "tour": "ATP Doubles",
                "tier": "Other",
                "tournament": "Test Doubles",
                "commence_time": "2026-07-19T18:00:00Z",
            }
        ]
    )
    result, _ = _prepare_moneyline_markets(
        markets,
        now=pd.Timestamp("2026-07-19T12:00:00Z"),
    )
    assert len(result) == 1
    assert not bool(result.iloc[0]["model_supported"])
    assert "singles" in str(result.iloc[0]["unsupported_reason"]).lower()


def test_order_book_refresh_updates_board(monkeypatch):
    original = pd.DataFrame(
        [
            {
                "market_id": "m1",
                "token_id1": "a",
                "token_id2": "b",
                "best_ask1": 0.55,
                "best_ask2": 0.46,
            }
        ]
    )
    board = SimpleNamespace(
        markets=original.copy(),
        matches=original.copy(),
        quotes=original.copy(),
    )

    class FakeClient:
        def enrich_order_books(self, frame):
            result = frame.copy()
            result["best_ask1"] = 0.57
            result["best_ask2"] = 0.44
            return result

    monkeypatch.setattr(pages, "PolymarketClient", FakeClient)
    updated = pages._refresh_polymarket_order_books(board)
    assert float(updated.markets.iloc[0]["best_ask1"]) == 0.57
    assert "price_snapshot_time" in updated.markets.columns
    assert updated.matches.equals(updated.markets)
    assert updated.quotes.equals(updated.markets)
