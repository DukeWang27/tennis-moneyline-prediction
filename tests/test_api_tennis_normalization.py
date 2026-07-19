from pathlib import Path

import pandas as pd

from tennis_predictor.data.api_tennis import (
    ProviderPlayerResolver,
    TournamentRegistry,
    normalize_api_tennis_fixtures,
)
from tennis_predictor.data.names import NameResolver


def test_api_tennis_uses_existing_surface_and_player_ids(tmp_path: Path):
    historical = pd.DataFrame(
        [
            {
                "tour": "ATP",
                "tournament": "Wimbledon",
                "surface": "Grass",
                "tier": "Slam",
                "player1": "Carlos Alcaraz",
                "player2": "Jannik Sinner",
                "player1_id": "ATP:1",
                "player2_id": "ATP:2",
            }
        ]
    )
    registry = TournamentRegistry(tmp_path / "tournaments.csv", historical)
    resolver = ProviderPlayerResolver(
        tmp_path / "aliases.csv", NameResolver.from_matches(historical)
    )
    raw = [
        {
            "event_key": "99",
            "event_date": "2026-07-01",
            "event_time": "11:00",
            "event_first_player": "C. Alcaraz",
            "first_player_key": "100",
            "event_second_player": "J. Sinner",
            "second_player_key": "200",
            "event_final_result": "3 - 1",
            "event_winner": "First Player",
            "event_status": "Finished",
            "event_type_type": "Atp Singles",
            "tournament_name": "Wimbledon",
            "tournament_key": "300",
            "tournament_round": "Quarter-finals",
        }
    ]
    normalized, unresolved = normalize_api_tennis_fixtures(
        raw,
        registry=registry,
        player_resolver=resolver,
        completed_only=True,
        randomize_completed_sides=False,
    )
    assert unresolved.empty
    row = normalized.iloc[0]
    assert row["surface"] == "Grass"
    assert row["tier"] == "Slam"
    assert row["player1_id"] == "ATP:1"
    assert row["player2_id"] == "ATP:2"
    assert row["fixture_id"] == "api_tennis:99"
