from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


KEEP_PREDICTION_COLUMNS = [
    "match_id", "date", "tour", "tier", "surface", "tournament", "round",
    "player1", "player2", "actual", "probability", "model", "fold",
    "eligible_with_filters", "any_filter_flag", "dead_rubber", "already_qualified",
    "fatigue_player1", "fatigue_player2", "pre_slam_rest_risk_player1",
    "pre_slam_rest_risk_player2", "large_line_move", "retirement", "walkover",
    "bet_side", "flat_stake", "flat_profit", "kelly_stake", "kelly_profit",
    "clv_probability", "clv_price",
]


def prepare(source: Path, destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for filename in ("trained_system.joblib", "metrics.csv"):
        src = source / filename
        if not src.exists():
            raise FileNotFoundError(f"Missing required artifact: {src}")
        target = destination / filename
        shutil.copy2(src, target)
        written.append(target)

    predictions_path = source / "predictions.parquet"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing required artifact: {predictions_path}")
    predictions = pd.read_parquet(predictions_path)
    columns = [column for column in KEEP_PREDICTION_COLUMNS if column in predictions.columns]
    compact = predictions[columns].copy()
    target = destination / "predictions.parquet"
    compact.to_parquet(target, index=False, compression="zstd")
    written.append(target)

    oversized = [path for path in written if path.stat().st_size > 95 * 1024 * 1024]
    if oversized:
        files = ", ".join(f"{path.name} ({path.stat().st_size / 1024**2:.1f} MB)" for path in oversized)
        raise RuntimeError(
            "Dashboard artifacts were created, but these files are close to or above GitHub's "
            f"100 MB limit: {files}. Store them in release/object storage instead of normal Git."
        )
    return written


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Create compact Streamlit dashboard artifacts.")
    parser.add_argument("--source", default="outputs/backtest")
    parser.add_argument("--destination", default="dashboard_artifacts")
    args = parser.parse_args(argv)
    written = prepare(Path(args.source), Path(args.destination))
    print("Prepared dashboard artifacts:")
    for path in written:
        print(f"- {path} ({path.stat().st_size / 1024**2:.2f} MB)")


if __name__ == "__main__":
    main()
