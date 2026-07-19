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
from tennis_predictor.services import backfill_completed_matches, rebuild_live_state


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


def sync_main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Backfill completed API-Tennis matches and update player state.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", default="outputs/backtest/trained_system.joblib")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    args = parser.parse_args(argv)
    config = _config(args.config)
    report = backfill_completed_matches(
        config,
        model_path=args.model,
        start_date=args.start,
        end_date=args.end,
    )
    print(json.dumps({
        "start_date": report.start_date,
        "end_date": report.end_date,
        "fetched": report.fetched,
        "appended": report.appended,
        "state_rows": report.state_rows,
        "unresolved": report.unresolved,
        "history_rows": report.history_rows,
        "ledger_rows": report.ledger_rows,
        "settled": report.settled,
    }, indent=2))


def rebuild_live_main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild post-cutoff state from a frozen pre-live model."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--live-model", default="outputs/backtest/trained_system.joblib")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", default=None)
    args = parser.parse_args(argv)
    config = _config(args.config)
    report = rebuild_live_state(
        config,
        base_model_path=args.base_model,
        live_model_path=args.live_model,
        start_date=args.start,
        end_date=args.end,
    )
    print(json.dumps({
        "start_date": report.start_date,
        "end_date": report.end_date,
        "fetched": report.fetched,
        "appended": report.appended,
        "state_rows": report.state_rows,
        "history_rows": report.history_rows,
        "ledger_rows": report.ledger_rows,
        "unresolved": report.unresolved,
        "settled": report.settled,
    }, indent=2))


def backfill_history_main(argv=None) -> None:
    """Download raw API-Tennis history, then normalize it into ID-first parquet."""
    from tennis_predictor.data import (
        ApiTennisHistoryParser,
        ApiTennisProfileHydrator,
        ApiTennisRawDownloader,
    )
    from tennis_predictor.storage import IdentityStore

    parser = argparse.ArgumentParser(
        description="Archive and normalize API-Tennis history in seven-day windows."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--parse-only", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--hydrate-profiles", action="store_true")
    parser.add_argument("--max-profiles", type=int, default=None)
    args = parser.parse_args(argv)
    if args.download_only and args.parse_only:
        parser.error("--download-only and --parse-only cannot be used together")

    config = _config(args.config)
    history = config.history
    start = args.start or history.start_date
    end = args.end or pd.Timestamp.today().date().isoformat()
    output: dict[str, object] = {"start_date": start, "end_date": end}

    if not args.parse_only:
        downloader = ApiTennisRawDownloader(
            api_key_env=config.live_data.api_tennis_key_env,
            timeout_seconds=config.data.request_timeout_seconds,
            max_retries=history.max_retries,
            retry_backoff_seconds=history.retry_backoff_seconds,
            sleep_between_requests=history.sleep_between_requests,
        )
        download_report = downloader.download(
            start_date=start,
            end_date=end,
            raw_root=history.raw_api_tennis_dir,
            timezone_name=config.live_data.timezone,
            force=args.force,
            continue_on_error=args.continue_on_error,
        )
        output["download"] = download_report.to_dict()

    if not args.download_only:
        identity_store = IdentityStore(history.identity_database_path)
        parser_service = ApiTennisHistoryParser(identity_store)
        matches, unresolved, parse_report = parser_service.parse_archive(
            history.raw_api_tennis_dir
        )
        parser_service.save(
            matches,
            unresolved,
            normalized_path=history.normalized_api_tennis_path,
            unresolved_path=history.unresolved_api_tennis_path,
        )
        output["parse"] = parse_report.to_dict()
        output["normalized_path"] = history.normalized_api_tennis_path
        output["unresolved_path"] = history.unresolved_api_tennis_path
        output["identity_database_path"] = history.identity_database_path

        if args.hydrate_profiles and not matches.empty:
            player_ids = pd.concat(
                [matches["player1_provider_id"], matches["player2_provider_id"]],
                ignore_index=True,
            ).dropna().astype(int)
            hydrated = ApiTennisProfileHydrator(
                identity_store,
                raw_root=history.raw_api_tennis_dir,
                api_key_env=config.live_data.api_tennis_key_env,
                timeout_seconds=config.data.request_timeout_seconds,
            ).hydrate(player_ids, max_profiles=args.max_profiles)
            output["hydrated_profiles"] = hydrated

    print(json.dumps(output, indent=2))



def kaggle_sync_main(argv=None) -> None:
    """Archive, validate, and normalize the configured Kaggle datasets."""
    from tennis_predictor.data import (
        KaggleCliClient,
        KaggleSyncService,
        default_dataset_definitions,
    )

    parser = argparse.ArgumentParser(
        description="Version and normalize ATP/WTA daily and Huge Tennis Kaggle data."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--source",
        action="append",
        choices=["all", "atp", "wta", "huge"],
        default=None,
        help="Repeat to select sources; default is all.",
    )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-ingest", action="store_true")
    parser.add_argument("--accept-schema-change", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args(argv)

    config = _config(args.config)
    aliases = {
        "atp": "kaggle_atp_daily",
        "wta": "kaggle_wta_daily",
        "huge": "huge_tennis",
    }
    requested = set(args.source or ["all"])
    selected = None if "all" in requested else {aliases[item] for item in requested}
    service = KaggleSyncService(
        identity_database_path=config.history.identity_database_path,
        raw_root=config.kaggle.raw_root,
        normalized_root=config.kaggle.normalized_root,
        cli_executable=config.kaggle.cli_executable,
        timeout_seconds=config.kaggle.command_timeout_seconds,
    )
    results = service.sync_all(
        default_dataset_definitions(config.kaggle),
        selected_sources=selected,
        force_download=args.force_download,
        force_ingest=args.force_ingest,
        accept_schema_change=(
            args.accept_schema_change or config.kaggle.accept_schema_change
        ),
        continue_on_error=args.continue_on_error,
    )
    print(json.dumps([result.to_dict() for result in results], indent=2))


def rebuild_candidate_main(argv=None) -> None:
    """Build, validate, and optionally promote an ID-first candidate model."""
    from tennis_predictor.services import build_candidate_model

    parser = argparse.ArgumentParser(
        description="Merge canonical sources, replay chronologically, and train a candidate model."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--api-matches", default=None)
    parser.add_argument("--existing-matches", default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--production-predictions",
        default="outputs/backtest/predictions.parquet",
    )
    parser.add_argument(
        "--production-model",
        default="outputs/backtest/trained_system.joblib",
    )
    parser.add_argument("--promote-if-passed", action="store_true")
    parser.add_argument("--skip-kaggle", action="store_true")
    args = parser.parse_args(argv)

    config = _config(args.config)
    history = config.history
    report = build_candidate_model(
        config,
        api_matches_path=args.api_matches or history.normalized_api_tennis_path,
        existing_matches_path=args.existing_matches or config.data.processed_path,
        start_date=args.start or history.start_date,
        end_date=args.end,
        output_dir=args.output_dir or history.candidate_output_dir,
        identity_database_path=history.identity_database_path,
        production_predictions_path=args.production_predictions,
        production_model_path=args.production_model,
        promote_if_passed=args.promote_if_passed,
        include_kaggle=not args.skip_kaggle,
    )
    print(json.dumps(report.to_dict(), indent=2))
