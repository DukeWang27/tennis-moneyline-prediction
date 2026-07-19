from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from tennis_predictor.config import AppConfig
from tennis_predictor.data.api_tennis import TournamentRegistry
from tennis_predictor.data.surface_audit import run_surface_audit
from tennis_predictor.storage.live_store import (
    initialize_live_store,
    resolve_tournament_metadata,
    save_tournament_metadata,
)


def test_registry_resolves_surface_by_tournament_edition(tmp_path: Path):
    registry_path = tmp_path / "tournaments.csv"
    pd.DataFrame(
        [
            {
                "provider": "official",
                "tournament_key": "",
                "tournament_name": "Test Open",
                "normalized_tournament": "test open",
                "tour": "WTA",
                "season": 2025,
                "surface": "Hard",
                "court": "Outdoor",
                "tier": "ITF",
                "confidence": "VERIFIED",
            },
            {
                "provider": "official",
                "tournament_key": "",
                "tournament_name": "Test Open",
                "normalized_tournament": "test open",
                "tour": "WTA",
                "season": 2026,
                "surface": "Clay",
                "court": "Outdoor",
                "tier": "ITF",
                "confidence": "VERIFIED",
            },
        ]
    ).to_csv(registry_path, index=False)

    registry = TournamentRegistry(registry_path, pd.DataFrame())
    result_2025 = registry.resolve_name(
        "Test Open", tour="WTA", match_date="2025-07-19", default_tier="ITF"
    )
    result_2026 = registry.resolve_name(
        "Test Open", tour="WTA", match_date="2026-07-19", default_tier="ITF"
    )
    unknown_year = registry.resolve_name(
        "Test Open", tour="WTA", match_date="2027-07-19", default_tier="ITF"
    )

    assert result_2025["surface"] == "Hard"
    assert result_2026["surface"] == "Clay"
    assert unknown_year["surface"] is None


def test_surface_audit_enriches_itf_women_by_name_and_season(tmp_path: Path):
    source_path = tmp_path / "clean_history.csv"
    pd.DataFrame(
        [
            {
                "source": "clean_wta",
                "date": "2026-07-01",
                "tour": "WTA",
                "competition_segment": "ITF_WOMEN",
                "tier": "ITF",
                "tournament": "W15 Test City",
                "surface": "Clay",
                "court": "Outdoor",
                "player1_id": "WTA:1",
                "player2_id": "WTA:2",
                "match_id": "clean:1",
            }
        ]
    ).to_csv(source_path, index=False)

    api_path = tmp_path / "api_matches.parquet"
    pd.DataFrame(
        [
            {
                "source": "api_tennis",
                "date": pd.Timestamp("2026-07-02"),
                "tour": "WTA",
                "competition_segment": "ITF_WOMEN",
                "tournament_level_raw": "Itf Women Singles",
                "tier": "ITF",
                "tournament": "W15 Test City",
                "provider_tournament_id": "900",
                "surface": "Unknown",
                "player1_provider_id": 10,
                "player2_provider_id": 20,
                "match_id": "api:1",
            }
        ]
    ).to_parquet(api_path, index=False)

    config = AppConfig()
    config.data.processed_path = str(tmp_path / "missing_processed.parquet")
    config.history.normalized_api_tennis_path = str(api_path)
    config.history.merged_matches_path = str(tmp_path / "missing_merged.parquet")
    config.kaggle.normalized_root = str(tmp_path / "missing_kaggle")
    config.live_data.tournament_registry_path = str(tmp_path / "tournaments.csv")

    report = run_surface_audit(
        config,
        extra_sources=[source_path],
        output_dir=tmp_path / "audit",
    )

    enriched = pd.read_parquet(report.api_enriched_path)
    inventory = json.loads((tmp_path / "audit" / "itf_women_inventory.json").read_text())
    assert enriched.loc[0, "surface"] == "Clay"
    assert enriched.loc[0, "court"] == "Outdoor"
    assert bool(enriched.loc[0, "surface_known"])
    assert inventory["rows"] == 1
    assert inventory["players"] == 2
    assert report.itf_women_known_surface_rows == 1


def test_live_store_migrates_legacy_metadata_and_saves_seasons(tmp_path: Path):
    db_path = tmp_path / "paper.db"
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE tournament_metadata_live (
            provider TEXT NOT NULL,
            provider_tournament TEXT NOT NULL,
            normalized_tournament TEXT NOT NULL,
            tour TEXT NOT NULL DEFAULT '',
            surface TEXT NOT NULL,
            tier TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(provider, normalized_tournament, tour)
        )
        """
    )
    connection.execute(
        "INSERT INTO tournament_metadata_live VALUES (?,?,?,?,?,?,?)",
        ("polymarket", "Legacy Open", "legacy open", "WTA", "Hard", "ITF", "2026-01-01"),
    )
    connection.commit()
    connection.close()

    initialize_live_store(db_path)
    legacy = resolve_tournament_metadata(
        db_path,
        provider="polymarket",
        tournament="Legacy Open",
        tour="WTA",
        match_date="2026-07-01",
    )
    assert legacy is not None
    assert legacy["surface"] == "Hard"
    assert legacy["court"] == "Unknown"

    save_tournament_metadata(
        db_path,
        provider="polymarket",
        tournament="Changing Open",
        tour="WTA",
        match_date="2025-07-01",
        surface="Hard",
        court="Outdoor",
        tier="ITF",
    )
    save_tournament_metadata(
        db_path,
        provider="polymarket",
        tournament="Changing Open",
        tour="WTA",
        match_date="2026-07-01",
        surface="Clay",
        court="Outdoor",
        tier="ITF",
    )

    assert resolve_tournament_metadata(
        db_path,
        provider="polymarket",
        tournament="Changing Open",
        tour="WTA",
        match_date="2025-07-19",
    )["surface"] == "Hard"
    assert resolve_tournament_metadata(
        db_path,
        provider="polymarket",
        tournament="Changing Open",
        tour="WTA",
        match_date="2026-07-19",
    )["surface"] == "Clay"
    assert resolve_tournament_metadata(
        db_path,
        provider="polymarket",
        tournament="Changing Open",
        tour="WTA",
        match_date="2027-07-19",
    ) is None


def test_surface_audit_links_prefixed_and_sponsored_tournament_names(tmp_path: Path):
    known_path = tmp_path / "known.csv"
    pd.DataFrame(
        [
            {
                "source": "clean_wta",
                "date": "2026-06-29",
                "tour": "WTA",
                "competition_segment": "WTA_MAIN",
                "tier": "Slam",
                "tournament": "Wimbledon",
                "surface": "Grass",
                "court": "Outdoor",
                "player1": "A Player",
                "player2": "B Player",
                "match_id": "known:1",
            },
            {
                "source": "clean_wta",
                "date": "2026-07-20",
                "tour": "WTA",
                "competition_segment": "WTA_MAIN",
                "tier": "WTA Tour",
                "tournament": "Prague Open",
                "surface": "Hard",
                "court": "Outdoor",
                "player1": "C Player",
                "player2": "D Player",
                "match_id": "known:2",
            },
        ]
    ).to_csv(known_path, index=False)

    api_path = tmp_path / "api.parquet"
    pd.DataFrame(
        [
            {
                "source": "api_tennis",
                "date": pd.Timestamp("2026-07-01"),
                "tour": "WTA",
                "competition_segment": "WTA_MAIN",
                "tournament_level_raw": "Wta Singles",
                "tier": "Slam",
                "tournament": "WTA Wimbledon",
                "tournament_key": "1379",
                "surface": "Unknown",
                "player1": "A Player",
                "player2": "B Player",
                "match_id": "api:1",
            },
            {
                "source": "api_tennis",
                "date": pd.Timestamp("2026-07-20"),
                "tour": "WTA",
                "competition_segment": "WTA_MAIN",
                "tournament_level_raw": "Wta Singles",
                "tier": "WTA Tour",
                "tournament": "Livesport Prague Open, Qualification",
                "tournament_key": "999",
                "surface": "Unknown",
                "player1": "C Player",
                "player2": "D Player",
                "match_id": "api:2",
            },
        ]
    ).to_parquet(api_path, index=False)

    config = AppConfig()
    config.data.processed_path = str(tmp_path / "missing.parquet")
    config.history.normalized_api_tennis_path = str(api_path)
    config.history.merged_matches_path = str(tmp_path / "missing_merged.parquet")
    config.kaggle.normalized_root = str(tmp_path / "missing_kaggle")
    config.live_data.tournament_registry_path = str(tmp_path / "registry.csv")

    report = run_surface_audit(
        config,
        extra_sources=[known_path],
        output_dir=tmp_path / "audit",
    )
    enriched = pd.read_parquet(report.api_enriched_path)
    assert enriched.set_index("match_id").loc["api:1", "surface"] == "Grass"
    assert enriched.set_index("match_id").loc["api:2", "surface"] == "Hard"
    linkage = pd.read_csv(tmp_path / "audit" / "api_tournament_linkage.csv")
    assert int(linkage["accepted"].sum()) == 2


def test_itf_women_queue_is_written_for_unknown_editions(tmp_path: Path):
    api_path = tmp_path / "api.parquet"
    pd.DataFrame(
        [
            {
                "source": "api_tennis",
                "date": pd.Timestamp("2026-07-01"),
                "tour": "WTA",
                "competition_segment": "ITF_WOMEN",
                "tournament_level_raw": "Itf Women Singles",
                "tier": "ITF",
                "tournament": "ITF W15 Test City Women",
                "tournament_key": "777",
                "surface": "Unknown",
                "player1": "A Player",
                "player2": "B Player",
                "match_id": "api:1",
            }
        ]
    ).to_parquet(api_path, index=False)
    config = AppConfig()
    config.data.processed_path = str(tmp_path / "missing.parquet")
    config.history.normalized_api_tennis_path = str(api_path)
    config.history.merged_matches_path = str(tmp_path / "missing_merged.parquet")
    config.kaggle.normalized_root = str(tmp_path / "missing_kaggle")
    config.live_data.tournament_registry_path = str(tmp_path / "registry.csv")
    run_surface_audit(config, output_dir=tmp_path / "audit")
    queue = pd.read_csv(tmp_path / "audit" / "itf_women_surface_queue.csv")
    assert len(queue) == 1
    assert queue.loc[0, "tournament_key"] == 777
    assert queue.loc[0, "review_status"] == "NEEDS_VERIFICATION"
