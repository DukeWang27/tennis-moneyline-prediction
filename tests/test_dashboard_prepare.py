from pathlib import Path

import pandas as pd

from tennis_predictor.ui.prepare import prepare


def test_prepare_dashboard_artifacts(tmp_path: Path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    (source / "trained_system.joblib").write_bytes(b"model")
    pd.DataFrame([{"filter_view": "all", "segment": "boosted"}]).to_csv(
        source / "metrics.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "match_id": "m1",
                "date": pd.Timestamp("2025-01-01"),
                "tour": "ATP",
                "tier": "250",
                "surface": "Hard",
                "tournament": "Test",
                "round": "F",
                "player1": "A",
                "player2": "B",
                "actual": 1,
                "probability": 0.7,
                "model": "boosted",
                "fold": 0,
                "unused_large_column": "drop me",
            }
        ]
    ).to_parquet(source / "predictions.parquet", index=False)

    written = prepare(source, destination)

    assert {path.name for path in written} == {
        "trained_system.joblib",
        "metrics.csv",
        "predictions.parquet",
    }
    compact = pd.read_parquet(destination / "predictions.parquet")
    assert "unused_large_column" not in compact.columns
    assert compact.loc[0, "match_id"] == "m1"
