from types import SimpleNamespace

import pandas as pd

from tennis_predictor.config import AppConfig
from tennis_predictor.services import live_data


def test_live_board_keeps_only_selected_new_york_day(tmp_path, monkeypatch):
    config = AppConfig()
    config.data.processed_path = str(tmp_path / "missing.parquet")
    config.data.cache_dir = str(tmp_path / "raw")
    config.filters.context_dir = str(tmp_path / "context")
    config.live_data.tournament_registry_path = str(tmp_path / "tournaments.csv")
    config.live_data.player_aliases_path = str(tmp_path / "aliases.csv")
    config.live_data.unresolved_fixtures_path = str(tmp_path / "unresolved.csv")
    config.paper_trading.database_path = str(tmp_path / "paper_trading.db")
    config.live_data.timezone = "America/New_York"

    markets = pd.DataFrame(
        [
            {
                "market_id": "ny-july-15",
                "commence_time": "2026-07-16T02:00:00Z",
                "parse_status": "NEEDS MAPPING",
                "parse_reason": "test",
                "player1": "",
                "player2": "",
                "tour": "ATP",
                "tier": "Challenger",
                "tournament": "Lincoln",
                "display_title": "Lincoln match",
                "question": "Lincoln match",
            },
            {
                "market_id": "ny-july-16",
                "commence_time": "2026-07-16T15:00:00Z",
                "parse_status": "NEEDS MAPPING",
                "parse_reason": "test",
                "player1": "",
                "player2": "",
                "tour": "ATP",
                "tier": "Challenger",
                "tournament": "Lincoln",
                "display_title": "Later Lincoln match",
                "question": "Later Lincoln match",
            },
        ]
    )
    client = SimpleNamespace(
        fetch_active_tennis_markets=lambda **kwargs: markets.copy(),
        last_diagnostics={"gamma_pages_read": 1},
    )
    system = SimpleNamespace(
        name_resolver=None,
        feature_engineer=SimpleNamespace(state=SimpleNamespace(as_of=None)),
    )
    monkeypatch.setattr(live_data, "upsert_polymarket_markets", lambda *args: 0)

    board = live_data.build_live_board(
        config,
        system=system,
        board_date="2026-07-15",
        polymarket_client=client,
    )

    assert board.markets["market_id"].tolist() == ["ny-july-15"]
    assert board.markets.iloc[0]["commence_time"].date().isoformat() == "2026-07-15"
    assert board.diagnostics["selected_board_date"] == "2026-07-15"
