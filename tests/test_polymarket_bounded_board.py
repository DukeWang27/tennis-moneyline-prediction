from __future__ import annotations

from typing import Any

from tennis_predictor.market import PolymarketClient


class FakeResponse:
    def __init__(self, payload: Any, status_code: int = 200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def json(self) -> Any:
        return self.payload


def _market(market_id: str, start: str, token_prefix: str) -> dict[str, Any]:
    return {
        "id": market_id,
        "question": "Player A vs Player B winner?",
        "active": True,
        "closed": False,
        "outcomes": '["Player A", "Player B"]',
        "outcomePrices": '["0.55", "0.45"]',
        "clobTokenIds": f'["{token_prefix}-a", "{token_prefix}-b"]',
        "events": [
            {
                "id": f"event-{market_id}",
                "title": "ITF Tennis: Player A vs Player B",
                "slug": market_id,
                "startDate": start,
            }
        ],
    }


def test_board_filters_date_before_batching_books(monkeypatch):
    in_day = _market("in-day", "2026-07-16T18:00:00Z", "in")
    out_day = _market("out-day", "2026-07-18T18:00:00Z", "out")
    posted: list[list[dict[str, str]]] = []

    def fake_get(url: str, params=None, timeout=None):
        if url.endswith("/markets"):
            return FakeResponse([in_day, out_day])
        if url.endswith("/public-search"):
            return FakeResponse({"events": []})
        raise AssertionError(url)

    def fake_post(url: str, json=None, timeout=None, headers=None):
        posted.append(json)
        books = []
        for item in json:
            books.append(
                {
                    "asset_id": item["token_id"],
                    "bids": [{"price": "0.50", "size": "5"}],
                    "asks": [{"price": "0.56", "size": "5"}],
                }
            )
        return FakeResponse(books)

    monkeypatch.setattr("requests.get", fake_get)
    monkeypatch.setattr("requests.post", fake_post)
    frame = PolymarketClient().fetch_active_tennis_markets(
        include_books=True,
        start_date="2026-07-16T04:00:00Z",
        end_date="2026-07-17T04:00:00Z",
        max_pages=4,
        max_markets=10,
    )

    assert frame["market_id"].tolist() == ["in-day"]
    assert len(posted) == 1
    assert {item["token_id"] for item in posted[0]} == {"in-a", "in-b"}
    assert frame.iloc[0].best_ask1 == 0.56
