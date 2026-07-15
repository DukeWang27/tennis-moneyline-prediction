from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from tennis_predictor.backtest import WalkForwardBacktester, print_summary
from tennis_predictor.config import AppConfig
from tennis_predictor.data import OddsAligner, SackmannLoader, load_normalized_snapshots
from tennis_predictor.features import FeatureEngineer
from tennis_predictor.filters import FilterEngine
from tennis_predictor.pipeline import TennisPredictionSystem


def _config(path: str) -> AppConfig:
    config = AppConfig.from_yaml(path)
    config.ensure_directories()
    return config


def download_main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Download and preprocess ATP/WTA matches.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    config = _config(args.config)
    loader = SackmannLoader(config.data)
    matches = loader.load(download=True, force_download=args.force)
    Path(config.data.processed_path).parent.mkdir(parents=True, exist_ok=True)
    matches.to_parquet(config.data.processed_path, index=False)
    print(f"Wrote {len(matches):,} matches to {config.data.processed_path}")


def backtest_main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run leakage-safe tennis backtest.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default="outputs/backtest")
    parser.add_argument("--redownload", action="store_true")
    args = parser.parse_args(argv)
    config = _config(args.config)
    processed = Path(config.data.processed_path)
    if processed.exists() and not args.redownload:
        matches = pd.read_parquet(processed)
    else:
        matches = SackmannLoader(config.data).load(download=True, force_download=args.redownload)
        processed.parent.mkdir(parents=True, exist_ok=True)
        matches.to_parquet(processed, index=False)

    matches = FilterEngine(config.filters).add_static_flags(matches)
    if config.odds.normalized_snapshots_path:
        snapshots = load_normalized_snapshots(config.odds.normalized_snapshots_path)
        matches, aligned = OddsAligner().align(matches, snapshots)
    else:
        aligned = pd.DataFrame()

    engineer = FeatureEngineer(config.features, config.filters)
    features = engineer.transform(matches, reset=True)
    backtester = WalkForwardBacktester(config.model, config.backtest)
    predictions, metrics = backtester.run(features)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output / "features.parquet", index=False)
    predictions.to_parquet(output / "predictions.parquet", index=False)
    metrics.to_csv(output / "metrics.csv", index=False)
    if not aligned.empty:
        aligned.to_parquet(output / "aligned_odds.parquet", index=False)
    system = TennisPredictionSystem(config).fit(matches)
    system.save(output / "trained_system.joblib")
    print_summary(metrics)
    print(f"\nSaved outputs to {output}")


def predict_main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Predict one tennis match.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--player1", required=True)
    parser.add_argument("--player2", required=True)
    parser.add_argument("--surface", required=True)
    parser.add_argument("--tier", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--tour", choices=["ATP", "WTA"], default=None)
    parser.add_argument("--dead-rubber", action="store_true")
    parser.add_argument("--already-qualified", action="store_true")
    parser.add_argument("--large-line-move", action="store_true")
    args = parser.parse_args(argv)
    system = TennisPredictionSystem.load(args.model)
    result = system.predict_match(
        player1=args.player1,
        player2=args.player2,
        surface=args.surface,
        tournament_tier=args.tier,
        match_date=args.date,
        tour=args.tour,
        dead_rubber=True if args.dead_rubber else None,
        already_qualified=True if args.already_qualified else None,
        large_line_move=args.large_line_move,
    )
    print(json.dumps(result, indent=2))
