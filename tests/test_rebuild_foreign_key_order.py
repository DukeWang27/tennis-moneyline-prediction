from types import SimpleNamespace

import pandas as pd

from tennis_predictor.config import AppConfig
from tennis_predictor.services import live_data


class FakeSystem:
    def __init__(self):
        self.feature_engineer = SimpleNamespace(
            state=SimpleNamespace(as_of=pd.Timestamp("2026-06-07"))
        )
        self.name_resolver = None

    def update_state(self, matches):
        return matches.copy()

    def save(self, path):
        return None


def test_rebuild_inserts_match_history_before_prediction_ledger(tmp_path, monkeypatch):
    config = AppConfig()
    config.data.processed_path = str(tmp_path / "matches.parquet")
    config.data.cache_dir = str(tmp_path / "raw")
    config.filters.context_dir = str(tmp_path / "context")
    config.live_data.tournament_registry_path = str(tmp_path / "tournaments.csv")
    config.live_data.player_aliases_path = str(tmp_path / "aliases.csv")
    config.live_data.unresolved_fixtures_path = str(tmp_path / "unresolved.csv")
    config.paper_trading.database_path = str(tmp_path / "paper_trading.db")

    pd.DataFrame(
        [{"match_id": "base", "date": pd.Timestamp("2026-06-07")}]
    ).to_parquet(config.data.processed_path, index=False)

    normalized = pd.DataFrame(
        [
            {
                "match_id": "new-1",
                "fixture_id": "new-1",
                "date": pd.Timestamp("2026-06-08"),
                "player1": "Player A",
                "player2": "Player B",
                "surface": "Unknown",
                "player1_won": 1.0,
            }
        ]
    )
    ledger = pd.DataFrame(
        [
            {
                "match_id": "new-1",
                "predicted_at": pd.Timestamp("2026-06-08"),
                "model_as_of": pd.Timestamp("2026-06-08"),
                "player1": "Player A",
                "player2": "Player B",
                "p_model1": None,
                "p_elo1": None,
                "actual": 1.0,
                "prediction_status": "NEEDS_SURFACE",
            }
        ]
    )

    monkeypatch.setattr(live_data.TennisPredictionSystem, "load", lambda _: FakeSystem())
    monkeypatch.setattr(live_data, "TournamentRegistry", lambda *args, **kwargs: object())
    monkeypatch.setattr(live_data, "ProviderPlayerResolver", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        live_data,
        "normalize_api_tennis_settlements",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    monkeypatch.setattr(
        live_data,
        "normalize_api_tennis_fixtures",
        lambda *args, **kwargs: (normalized, pd.DataFrame()),
    )
    monkeypatch.setattr(
        live_data,
        "settle_pending_bets_from_results",
        lambda *args, **kwargs: {"won": 0, "lost": 0, "void": 0},
    )
    monkeypatch.setattr(live_data, "_feature_predictions", lambda *args: ledger)

    order = []
    monkeypatch.setattr(
        live_data,
        "upsert_match_history",
        lambda *args, **kwargs: order.append("history") or 1,
    )
    monkeypatch.setattr(
        live_data,
        "upsert_prediction_ledger",
        lambda *args, **kwargs: order.append("ledger") or 1,
    )

    client = SimpleNamespace(fetch_fixtures=lambda *args, **kwargs: pd.DataFrame([{}]))
    live_data.rebuild_live_state(
        config,
        base_model_path=tmp_path / "base.joblib",
        live_model_path=tmp_path / "live.joblib",
        start_date="2026-06-08",
        end_date="2026-06-08",
        client=client,
    )

    assert order == ["history", "ledger"]
