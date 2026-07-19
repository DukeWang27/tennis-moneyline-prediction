from pathlib import Path
from datetime import date

import pandas as pd

from tennis_predictor.data import (
    KaggleDatasetDefinition,
    KaggleRemoteFile,
    KaggleRemoteManifest,
    KaggleSyncService,
)


class FakeClient:
    def __init__(self):
        self.downloads = 0

    def list_files(self, slug):
        return KaggleRemoteManifest(
            slug=slug,
            files=[KaggleRemoteFile("matches.csv", "100", "2026-07-17")],
        )

    def download(self, slug, destination):
        self.downloads += 1
        pd.DataFrame(
            {
                "Date": ["2026-07-16"],
                "Tournament": ["Example Open"],
                "Winner": ["A Player"],
                "Loser": ["B Player"],
                "Surface": ["Hard"],
                "Round": ["R32"],
            }
        ).to_csv(Path(destination) / "matches.csv", index=False)


def test_sync_archives_and_skips_unchanged_manifest(tmp_path):
    service = KaggleSyncService(
        identity_database_path=tmp_path / "identity.db",
        raw_root=tmp_path / "raw",
        normalized_root=tmp_path / "normalized",
    )
    fake = FakeClient()
    service.client = fake
    definition = KaggleDatasetDefinition(
        "kaggle_atp_daily", "owner/atp", "atp_daily", True
    )
    first = service.sync_one(definition)
    second = service.sync_one(definition)
    assert first.status == "INGESTED"
    assert second.status == "SKIPPED_UNCHANGED_REMOTE_MANIFEST"
    assert fake.downloads == 1
    assert Path(first.archive_path).parent.name == date.today().isoformat()
    assert Path(first.normalized_path).exists()
