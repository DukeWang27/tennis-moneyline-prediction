from __future__ import annotations

from typing import Any

from tennis_predictor.market import PolymarketClient


class FakeResponse:
    def __init__(self, payload: Any):
        self.payload = payload
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self.payload


def test_active_refresh_uses_cursor_not_large_offsets(monkeypatch):
    calls: list[tuple[str, dict[str, Any]]] = []
    market = {
        "id": "tennis-keyset-1",
        "question": "Player A vs Player B winner?",
        "active": True,
        "closed": False,
        "outcomes": '["Player A", "Player B"]',
        "outcomePrices": '["0.55", "0.45"]',
        "clobTokenIds": '["a", "b"]',
        "events": [
            {
                "id": "event-1",
                "title": "ATP Challenger Lincoln Tennis: Player A vs Player B",
                "slug": "player-a-player-b",
                "startDate": "2026-07-16T18:00:00Z",
            }
        ],
    }

    def fake_get(url: str, params=None, timeout=None):
        params = params or {}
        calls.append((url, params))
        if url.endswith("/markets"):
            return FakeResponse([])
        if url.endswith("/public-search"):
            return FakeResponse({"events": []})
        if url.endswith("/markets/keyset"):
            if params.get("after_cursor") == "cursor-1":
                return FakeResponse({"markets": []})
            return FakeResponse({"markets": [market], "next_cursor": "cursor-1"})
        raise AssertionError(url)

    monkeypatch.setattr("requests.get", fake_get)
    frame = PolymarketClient().fetch_active_tennis_markets(
        include_books=False,
        page_size=100,
        max_pages=3,
        start_date="2026-07-16T04:00:00Z",
        end_date="2026-07-17T04:00:00Z",
    )

    assert len(frame) == 1
    keyset_calls = [(url, params) for url, params in calls if url.endswith("/markets/keyset")]
    assert len(keyset_calls) == 2
    assert all("offset" not in params for _, params in keyset_calls)
    assert keyset_calls[1][1]["after_cursor"] == "cursor-1"
    assert keyset_calls[0][1]["start_date_min"] == "2026-07-16T04:00:00+00:00"
    assert keyset_calls[0][1]["start_date_max"] == "2026-07-17T04:00:00+00:00"
