from types import SimpleNamespace

from tennis_predictor.config import AppConfig
from tennis_predictor.services import live_data


class FakeIdentityStore:
    def __init__(self, database_path):
        self.database_path = database_path

    def resolve_name(self, *args, **kwargs):
        return SimpleNamespace(
            resolved=True,
            provider_player_id=None,
        )

    def model_player_id(self, provider_player_id):
        raise AssertionError(
            "model_player_id must not be called when "
            "provider_player_id is None"
        )


class FakeResolver:
    def __init__(self):
        self.by_id = {}

    def resolve(self, *args, **kwargs):
        raise KeyError("not found")


def test_resolved_name_without_provider_id_uses_fallback(
    tmp_path,
    monkeypatch,
):
    config = AppConfig()

    config.history.identity_database_path = str(
        tmp_path / "identity.db"
    )

    config.paper_trading.database_path = str(
        tmp_path / "paper.db"
    )

    system = SimpleNamespace(
        name_resolver=FakeResolver(),
    )

    monkeypatch.setattr(
        live_data,
        "IdentityStore",
        FakeIdentityStore,
    )

    monkeypatch.setattr(
        live_data,
        "resolve_player_alias",
        lambda *args, **kwargs: None,
    )

    result = live_data._resolve_identity(
        config,
        system,
        "Example Player",
        "ATP",
    )

    assert result is None
