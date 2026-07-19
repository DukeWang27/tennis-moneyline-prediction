import sqlite3

import pandas as pd

from tennis_predictor.data.match_consolidation import CanonicalMatchConsolidator
from tennis_predictor.storage.identity_store import IdentityStore


def test_candidate_persist_copies_players_before_matches(tmp_path):
    identity_store = IdentityStore(tmp_path / "identity.db")

    player1_id = identity_store.create_player(
        canonical_name="Player One",
        tour="ATP",
    )
    player2_id = identity_store.create_player(
        canonical_name="Player Two",
        tour="ATP",
    )

    matches = pd.DataFrame(
        [
            {
                "match_fingerprint": "test-match-fingerprint",
                "chronology_ts": pd.Timestamp("2026-01-01 12:00:00+00:00"),
                "date": pd.Timestamp("2026-01-01"),
                "player1_canonical_id": player1_id,
                "player2_canonical_id": player2_id,
                "tournament": "Test Tournament",
                "round": "R32",
                "tour": "ATP",
                "tier": "Tour",
                "surface": "Hard",
                "player1_won": 1,
                "score": "6-4 6-4",
                "status": "Completed",
                "retirement": False,
                "walkover": False,
                "defaulted": False,
                "void": False,
                "sources_present": '["test"]',
                "field_provenance": "{}",
            }
        ]
    )

    source_records = pd.DataFrame(
        columns=[
            "source_name",
            "source_match_id",
            "match_fingerprint",
            "raw_record_json",
        ]
    )

    candidate_database = tmp_path / "candidate.db"

    CanonicalMatchConsolidator(identity_store).persist(
        matches,
        source_records,
        database_path=candidate_database,
        parquet_path=tmp_path / "matches.parquet",
    )

    with sqlite3.connect(candidate_database) as connection:
        player_count = connection.execute(
            "SELECT COUNT(*) FROM players"
        ).fetchone()[0]

        match_count = connection.execute(
            "SELECT COUNT(*) FROM canonical_matches"
        ).fetchone()[0]

        foreign_key_errors = connection.execute(
            "PRAGMA foreign_key_check"
        ).fetchall()

    assert player_count == 2
    assert match_count == 1
    assert foreign_key_errors == []
