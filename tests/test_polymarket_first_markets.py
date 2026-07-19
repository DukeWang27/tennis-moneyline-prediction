from typing import Any

from tennis_predictor.market.live_odds import PolymarketClient


class FakeResponse:
    def __init__(self, payload: Any):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_gamma_markets_are_rendered_even_when_player_parse_fails(monkeypatch):
    calls = []
    tennis_market = {
        "id": "tennis-1",
        "question": "Will the Lincoln Challenger tennis match finish today?",
        "active": True,
        "closed": False,
        "outcomes": '["Yes", "No"]',
        "outcomePrices": '["0.5", "0.5"]',
        "clobTokenIds": '["yes-token", "no-token"]',
        "events": [
            {
                "id": "event-1",
                "title": "ATP Challenger Lincoln Tennis",
                "slug": "challenger-lincoln-tennis",
                "startDate": "2026-07-20T18:00:00Z",
            }
        ],
    }

    def fake_get(url, params=None, timeout=None):
        calls.append((url, params or {}))
        if url.endswith("/markets"):
            return FakeResponse([tennis_market] if (params or {}).get("offset") == 0 else [])
        raise AssertionError(url)

    monkeypatch.setattr("requests.get", fake_get)
    frame = PolymarketClient().fetch_active_tennis_markets(
        include_books=False, page_size=100, max_pages=2
    )

    assert len(frame) == 1
    assert frame.iloc[0].market_id == "tennis-1"
    assert frame.iloc[0].parse_status == "NEEDS MAPPING"
    assert any(url.endswith("/markets") for url, _ in calls)
    assert calls[0][1]["active"] == "true"
    assert calls[0][1]["closed"] == "false"
