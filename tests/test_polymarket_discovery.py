from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from tennis_predictor.market.live_odds import PolymarketClient


class FakeResponse:
    def __init__(self, payload: Any, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> Any:
        return self._payload


def test_public_search_discovers_tennis_and_uses_executable_asks(monkeypatch):
    calls: list[tuple[str, dict[str, Any]]] = []
    event = {
        "id": "event-1",
        "slug": "player-a-player-b",
        "title": "ATP Bastad: Player A vs Player B",
        "active": True,
        "closed": False,
        "startDate": "2026-07-16T15:00:00Z",
        "seriesTitle": "ATP Bastad",
        "markets": [
            {
                "id": "market-1",
                "question": "Player A vs Player B winner?",
                "active": True,
                "closed": False,
                "outcomes": '["Player A", "Player B"]',
                "outcomePrices": '["0.55", "0.45"]',
                "clobTokenIds": '["token-a", "token-b"]',
                "liquidityNum": 500,
            }
        ],
    }

    def fake_get(url: str, params=None, timeout=None):
        params = params or {}
        calls.append((url, params))
        if url.endswith("/public-search"):
            payload = {"events": [event] if params.get("q") == "tennis" else []}
            return FakeResponse(payload)
        if url.endswith("/book"):
            if params["token_id"] == "token-a":
                return FakeResponse(
                    {
                        "bids": [{"price": "0.53", "size": "50"}],
                        "asks": [{"price": "0.56", "size": "40"}],
                    }
                )
            return FakeResponse(
                {
                    "bids": [{"price": "0.43", "size": "30"}],
                    "asks": [{"price": "0.46", "size": "20"}],
                }
            )
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr("requests.get", fake_get)
    client = PolymarketClient()
    frame = client.fetch_tennis_quotes(
        use_order_books=True,
        limit=10,
        search_pages=1,
        keyset_pages=1,
    )

    assert len(frame) == 1
    row = frame.iloc[0]
    assert row.player1 == "Player A"
    assert row.player2 == "Player B"
    assert row.polymarket_price1 == pytest.approx(0.56)
    assert row.polymarket_price2 == pytest.approx(0.46)
    assert row.decimal_odds1 == pytest.approx(1 / 0.56)
    assert row.best_bid1 == pytest.approx(0.53)
    assert row.spread1 == pytest.approx(0.03)
    assert row.market_url.endswith("/event/player-a-player-b")
    assert pd.Timestamp(row.commence_time) == pd.Timestamp("2026-07-16 15:00:00")
    assert not any(url.endswith("/sports") for url, _ in calls)
    assert client.last_diagnostics["parsed_quote_rows"] == 1


def test_empty_search_returns_stable_quote_columns(monkeypatch):
    def fake_get(url: str, params=None, timeout=None):
        if url.endswith("/public-search"):
            return FakeResponse({"events": []})
        if url.endswith("/markets/keyset"):
            return FakeResponse({"markets": []})
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr("requests.get", fake_get)
    frame = PolymarketClient().fetch_tennis_quotes(
        use_order_books=False,
        search_pages=1,
        keyset_pages=1,
    )
    assert frame.empty
    assert {"player1", "player2", "market_id", "token_id1"}.issubset(frame.columns)
