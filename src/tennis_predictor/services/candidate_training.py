from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from tennis_predictor.backtest import WalkForwardBacktester
from tennis_predictor.backtest.metrics import (
    calibration_curve_table,
    expected_calibration_error,
)
from tennis_predictor.config import AppConfig
from tennis_predictor.data.names import normalize_name
from tennis_predictor.data.match_consolidation import (
    CanonicalMatchConsolidator,
    derive_competition_segment,
)
from tennis_predictor.features import FeatureEngineer
from tennis_predictor.filters import FilterEngine
from tennis_predictor.pipeline import TennisPredictionSystem
from tennis_predictor.storage.identity_store import IdentityStore

_REQUIRED_GATE_SEGMENTS = ("ATP_CHALLENGER", "ITF_MEN", "ITF_WOMEN")


def _atomic_parquet(path: str | Path, frame: pd.DataFrame) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, destination)


def _safe_frame(path: str | Path | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    source = Path(path)
    if not source.exists():
        return pd.DataFrame()
    if source.suffix.casefold() in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    if source.suffix.casefold() == ".csv":
        return pd.read_csv(source, low_memory=False)
    raise ValueError(f"Unsupported candidate source: {source}")


def latest_kaggle_sources(config: AppConfig) -> list[tuple[pd.DataFrame, str]]:
    """Load only the latest accepted normalized snapshot for each Kaggle source."""
    root = Path(config.kaggle.normalized_root)
    definitions = (
        ("atp_daily", "kaggle_atp_daily"),
        ("wta_daily", "kaggle_wta_daily"),
        ("huge_tennis", "huge_tennis"),
    )
    sources: list[tuple[pd.DataFrame, str]] = []
    for archive_name, source_name in definitions:
        marker = root / archive_name / "LATEST.json"
        if not marker.exists():
            continue
        payload = json.loads(marker.read_text(encoding="utf-8"))
        normalized_path = payload.get("normalized_path")
        frame = _safe_frame(normalized_path)
        if not frame.empty:
            sources.append((frame, source_name))
    return sources


def _prepare_closing_odds(frame: pd.DataFrame) -> pd.DataFrame:
    """Expose closing prices to ROI code while keeping them out of ML features.

    The existing backtester expects open/close columns. Kaggle provides closing
    prices, so both execution and comparison columns are intentionally populated
    from the same closing quote and labelled as a closing-price benchmark. None
    of these columns are in ``FeatureEngineer.numeric_feature_columns()``.
    """
    output = frame.copy()
    odds1 = pd.to_numeric(
        output.get("closing_odds_player1", pd.Series(np.nan, index=output.index)),
        errors="coerce",
    )
    odds2 = pd.to_numeric(
        output.get("closing_odds_player2", pd.Series(np.nan, index=output.index)),
        errors="coerce",
    )
    valid = odds1.gt(1.0) & odds2.gt(1.0)
    inverse1 = 1.0 / odds1
    inverse2 = 1.0 / odds2
    overround = inverse1 + inverse2
    market_p1 = (inverse1 / overround).where(valid & overround.gt(0))
    for suffix in ("open", "close"):
        output[f"market_decimal1_{suffix}"] = odds1.where(valid)
        output[f"market_decimal2_{suffix}"] = odds2.where(valid)
        output[f"market_p1_{suffix}"] = market_p1
    output["odds_benchmark_type"] = np.where(valid, "historical_closing", pd.NA)
    return output


def prepare_chronological_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize consolidated matches before sequential replay."""
    if matches is None or matches.empty:
        return pd.DataFrame()
    output = matches.copy()
    output["date"] = pd.to_datetime(output["date"], errors="coerce").dt.tz_localize(None)
    output["chronology_ts"] = pd.to_datetime(
        output.get("chronology_ts", output["date"]), errors="coerce", utc=True
    ).dt.tz_convert(None)
    output = output.dropna(
        subset=["date", "chronology_ts", "player1_id", "player2_id", "player1_won"]
    ).copy()
    defaults: dict[str, object] = {
        "event_date": pd.NaT,
        "match_num": 0,
        "round": "Unknown",
        "tournament": "Unknown",
        "surface": "Unknown",
        "tier": "ITF",
        "tour": "",
        "retirement": False,
        "walkover": False,
        "defaulted": False,
        "void": False,
        "is_qualifying": False,
        "date_is_estimated": False,
        "player1_age": np.nan,
        "player2_age": np.nan,
        "player1_rank": np.nan,
        "player2_rank": np.nan,
    }
    for column, default in defaults.items():
        if column not in output:
            output[column] = default
        else:
            output[column] = output[column].fillna(default)
    output["event_date"] = pd.to_datetime(output["event_date"], errors="coerce").fillna(
        output["date"]
    )
    output["player1_won"] = pd.to_numeric(output["player1_won"], errors="coerce")
    output = output[output["player1_won"].isin([0, 1])].copy()
    output["competition_segment"] = derive_competition_segment(output)
    output = _prepare_closing_odds(output)
    output = output.sort_values(["chronology_ts", "match_id"], kind="mergesort").reset_index(
        drop=True
    )
    if not output["chronology_ts"].is_monotonic_increasing:
        raise AssertionError("Chronological match order is not monotonic.")
    return output


@dataclass(slots=True)
class AssemblyReport:
    input_rows: int
    canonical_rows: int
    unresolved_rows: int
    duplicate_rows_removed: int
    output_rows: int
    by_source: dict[str, int]
    by_segment: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MultiSourceMatchAssembler:
    """Compatibility wrapper around the canonical ID-first consolidator."""

    def __init__(
        self,
        identity_store: IdentityStore,
        *,
        fuzzy_threshold: float = 97.0,
        minimum_margin: float = 4.0,
    ):
        self.consolidator = CanonicalMatchConsolidator(
            identity_store,
            fuzzy_threshold=fuzzy_threshold,
            minimum_margin=minimum_margin,
        )

    def assemble(
        self,
        sources: list[tuple[pd.DataFrame, str]],
        *,
        start_date: object | None = None,
        end_date: object | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, AssemblyReport]:
        matches, unresolved, report, _ = self.consolidator.consolidate(
            sources, start_date=start_date, end_date=end_date
        )
        return matches, unresolved, AssemblyReport(
            input_rows=report.input_rows,
            canonical_rows=report.resolved_rows,
            unresolved_rows=report.unresolved_rows,
            duplicate_rows_removed=report.duplicate_source_rows_merged,
            output_rows=report.unique_matches,
            by_source=report.by_source,
            by_segment=report.by_segment,
        )


@dataclass(slots=True)
class ModelGateReport:
    status: str
    required_segments: list[str]
    passed_segments: list[str]
    failed_segments: list[str]
    inconclusive_segments: list[str]
    comparisons: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _boosted_segment_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics is None or metrics.empty:
        return pd.DataFrame()
    required = {"filter_view", "segment_type", "segment"}
    if not required.issubset(metrics.columns):
        return pd.DataFrame()
    return metrics[
        metrics["filter_view"].eq("all")
        & metrics["segment_type"].eq("competition_segment")
        & metrics["segment"].astype(str).str.startswith("boosted|")
    ].copy()


def evaluate_model_gate(
    candidate_metrics: pd.DataFrame,
    production_metrics: pd.DataFrame,
    *,
    required_segments: tuple[str, ...] = _REQUIRED_GATE_SEGMENTS,
    minimum_segment_matches: int = 200,
) -> ModelGateReport:
    """Non-statistical compatibility gate used for reports and older tests."""
    candidate_rows = _boosted_segment_rows(candidate_metrics)
    production_rows = _boosted_segment_rows(production_metrics)
    candidate = candidate_rows.set_index("segment") if not candidate_rows.empty else pd.DataFrame()
    production = production_rows.set_index("segment") if not production_rows.empty else pd.DataFrame()
    passed: list[str] = []
    failed: list[str] = []
    inconclusive: list[str] = []
    comparisons: list[dict[str, Any]] = []
    for segment in required_segments:
        key = f"boosted|{segment}"
        if key not in candidate.index or key not in production.index:
            inconclusive.append(segment)
            comparisons.append(
                {"segment": segment, "status": "INCONCLUSIVE", "reason": "missing metrics"}
            )
            continue
        cand = candidate.loc[key]
        prod = production.loc[key]
        if isinstance(cand, pd.DataFrame):
            cand = cand.iloc[0]
        if isinstance(prod, pd.DataFrame):
            prod = prod.iloc[0]
        if min(int(cand["matches"]), int(prod["matches"])) < minimum_segment_matches:
            inconclusive.append(segment)
            comparisons.append(
                {
                    "segment": segment,
                    "status": "INCONCLUSIVE",
                    "reason": f"fewer than {minimum_segment_matches} matches",
                }
            )
            continue
        is_pass = float(cand["log_loss"]) < float(prod["log_loss"]) and float(
            cand["ece"]
        ) < float(prod["ece"])
        (passed if is_pass else failed).append(segment)
        comparisons.append(
            {
                "segment": segment,
                "status": "PASS" if is_pass else "FAIL",
                "candidate_log_loss": float(cand["log_loss"]),
                "production_log_loss": float(prod["log_loss"]),
                "candidate_ece": float(cand["ece"]),
                "production_ece": float(prod["ece"]),
            }
        )
    status = "FAIL" if failed else "INCONCLUSIVE" if inconclusive else "PASS"
    return ModelGateReport(
        status,
        list(required_segments),
        passed,
        failed,
        inconclusive,
        comparisons,
    )


def _boosted_predictions(frame: pd.DataFrame, segment: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    output = frame.copy()
    if "competition_segment" not in output:
        output["competition_segment"] = derive_competition_segment(output)
    if "model" in output:
        output = output[output["model"].eq("boosted")]
    return output[output["competition_segment"].eq(segment)].copy()


def _legacy_pair_key(frame: pd.DataFrame) -> pd.DataFrame:
    """Create an orientation-neutral comparison key for legacy predictions."""
    required = {"date", "player1", "player2", "actual", "probability"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    output = frame.copy()
    name1 = output["player1"].fillna("").astype(str).map(normalize_name)
    name2 = output["player2"].fillna("").astype(str).map(normalize_name)
    first_is_anchor = name1.le(name2)
    anchor1 = name1.where(first_is_anchor, name2)
    anchor2 = name2.where(first_is_anchor, name1)
    date = pd.to_datetime(output["date"], errors="coerce").dt.date.astype("string")
    tournament = output.get("tournament", pd.Series("", index=output.index)).fillna("").astype(str).map(normalize_name)
    round_name = output.get("round", pd.Series("", index=output.index)).fillna("").astype(str).map(normalize_name)
    output["__paired_key"] = (
        date + "|" + anchor1 + "|" + anchor2 + "|" + tournament + "|" + round_name
    )
    actual = pd.to_numeric(output["actual"], errors="coerce")
    probability = pd.to_numeric(output["probability"], errors="coerce")
    output["__paired_actual"] = actual.where(first_is_anchor, 1.0 - actual)
    output["__paired_probability"] = probability.where(
        first_is_anchor, 1.0 - probability
    )
    return output


def _paired_views(
    candidate: pd.DataFrame, production: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    if (
        "match_fingerprint" in candidate
        and "match_fingerprint" in production
        and candidate["match_fingerprint"].notna().any()
        and production["match_fingerprint"].notna().any()
    ):
        left = candidate.copy()
        right = production.copy()
        for frame in (left, right):
            frame["__paired_key"] = frame["match_fingerprint"].astype(str)
            frame["__paired_actual"] = pd.to_numeric(frame["actual"], errors="coerce")
            frame["__paired_probability"] = pd.to_numeric(
                frame["probability"], errors="coerce"
            )
        return left, right
    left = _legacy_pair_key(candidate)
    right = _legacy_pair_key(production)
    if left.empty or right.empty:
        return None
    return left, right


def evaluate_paired_model_gate(
    candidate_predictions: pd.DataFrame,
    production_predictions: pd.DataFrame,
    *,
    required_segments: tuple[str, ...] = _REQUIRED_GATE_SEGMENTS,
    minimum_segment_matches: int = 200,
    bootstrap_samples: int = 2000,
    confidence: float = 0.95,
    random_state: int = 27,
) -> ModelGateReport:
    """Require paired bootstrap improvement in log loss and calibration.

    The gate compares predictions for the exact same matches. A candidate only
    passes a segment when the upper confidence bound for both candidate-minus-
    production log loss and ECE is below zero. Missing overlap is inconclusive,
    never an automatic promotion.
    """
    passed: list[str] = []
    failed: list[str] = []
    inconclusive: list[str] = []
    comparisons: list[dict[str, Any]] = []
    rng = np.random.default_rng(random_state)
    alpha = 1.0 - float(confidence)

    for segment in required_segments:
        candidate = _boosted_predictions(candidate_predictions, segment)
        production = _boosted_predictions(production_predictions, segment)
        views = _paired_views(candidate, production)
        if views is None:
            inconclusive.append(segment)
            comparisons.append(
                {"segment": segment, "status": "INCONCLUSIVE", "reason": "no shared match key"}
            )
            continue
        candidate_view, production_view = views
        left = candidate_view[["__paired_key", "__paired_actual", "__paired_probability"]].rename(
            columns={
                "__paired_actual": "actual_candidate",
                "__paired_probability": "candidate_probability",
            }
        )
        right = production_view[["__paired_key", "__paired_actual", "__paired_probability"]].rename(
            columns={
                "__paired_actual": "actual_production",
                "__paired_probability": "production_probability",
            }
        )
        paired = left.merge(right, on="__paired_key", how="inner").dropna()
        paired = paired[
            paired["actual_candidate"].astype(int).eq(paired["actual_production"].astype(int))
        ].drop_duplicates("__paired_key")
        if len(paired) < int(minimum_segment_matches):
            inconclusive.append(segment)
            comparisons.append(
                {
                    "segment": segment,
                    "status": "INCONCLUSIVE",
                    "reason": f"only {len(paired)} paired matches",
                    "required": int(minimum_segment_matches),
                }
            )
            continue

        y = paired["actual_candidate"].astype(int).to_numpy()
        cand_p = paired["candidate_probability"].clip(1e-6, 1 - 1e-6).to_numpy()
        prod_p = paired["production_probability"].clip(1e-6, 1 - 1e-6).to_numpy()
        observed_log_delta = float(log_loss(y, cand_p, labels=[0, 1]) - log_loss(y, prod_p, labels=[0, 1]))
        observed_ece_delta = float(
            expected_calibration_error(y, cand_p) - expected_calibration_error(y, prod_p)
        )
        log_deltas = np.empty(int(bootstrap_samples), dtype=float)
        ece_deltas = np.empty(int(bootstrap_samples), dtype=float)
        for index in range(int(bootstrap_samples)):
            sampled = rng.integers(0, len(y), size=len(y))
            sampled_y = y[sampled]
            log_deltas[index] = log_loss(
                sampled_y, cand_p[sampled], labels=[0, 1]
            ) - log_loss(sampled_y, prod_p[sampled], labels=[0, 1])
            ece_deltas[index] = expected_calibration_error(
                sampled_y, cand_p[sampled]
            ) - expected_calibration_error(sampled_y, prod_p[sampled])
        low_q, high_q = alpha / 2.0, 1.0 - alpha / 2.0
        log_ci = (float(np.quantile(log_deltas, low_q)), float(np.quantile(log_deltas, high_q)))
        ece_ci = (float(np.quantile(ece_deltas, low_q)), float(np.quantile(ece_deltas, high_q)))
        is_pass = log_ci[1] < 0.0 and ece_ci[1] < 0.0
        status = "PASS" if is_pass else "FAIL"
        (passed if is_pass else failed).append(segment)
        comparisons.append(
            {
                "segment": segment,
                "status": status,
                "paired_matches": len(paired),
                "candidate_minus_production_log_loss": observed_log_delta,
                "log_loss_delta_ci": list(log_ci),
                "candidate_minus_production_ece": observed_ece_delta,
                "ece_delta_ci": list(ece_ci),
                "confidence": confidence,
            }
        )

    status = "FAIL" if failed else "INCONCLUSIVE" if inconclusive else "PASS"
    return ModelGateReport(
        status,
        list(required_segments),
        passed,
        failed,
        inconclusive,
        comparisons,
    )


@dataclass(slots=True)
class CandidateBuildReport:
    matches_path: str
    features_path: str
    predictions_path: str
    metrics_path: str
    calibration_path: str
    candidate_model_path: str
    unresolved_path: str
    assembly: dict[str, Any]
    model_gate: dict[str, Any]
    promoted: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _candidate_sources(
    config: AppConfig,
    *,
    api_matches_path: str | Path | None,
    existing_matches_path: str | Path | None,
    include_kaggle: bool,
) -> list[tuple[pd.DataFrame, str]]:
    sources: list[tuple[pd.DataFrame, str]] = []
    for path, name in (
        (api_matches_path, "api_tennis"),
        (existing_matches_path, "sackmann"),
    ):
        frame = _safe_frame(path)
        if not frame.empty:
            sources.append((frame, name))
    if include_kaggle:
        sources.extend(latest_kaggle_sources(config))
    return sources


def build_candidate_model(
    config: AppConfig,
    *,
    api_matches_path: str | Path | None,
    existing_matches_path: str | Path | None,
    start_date: object,
    end_date: object | None,
    output_dir: str | Path,
    identity_database_path: str | Path,
    production_predictions_path: str | Path | None = None,
    production_model_path: str | Path | None = None,
    promote_if_passed: bool = False,
    include_kaggle: bool = True,
) -> CandidateBuildReport:
    """Consolidate all sources, replay chronologically, and train a candidate.

    Production model/data files are never overwritten unless the paired
    statistical gate returns PASS and ``promote_if_passed`` is explicitly set.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    identity_store = IdentityStore(identity_database_path)
    sources = _candidate_sources(
        config,
        api_matches_path=api_matches_path,
        existing_matches_path=existing_matches_path,
        include_kaggle=include_kaggle,
    )
    if not sources:
        raise RuntimeError("No source datasets were available for candidate training.")

    consolidator = CanonicalMatchConsolidator(
        identity_store,
        fuzzy_threshold=config.history.fuzzy_match_threshold,
        minimum_margin=config.history.fuzzy_minimum_margin,
    )
    matches, unresolved, consolidation_report, source_records = consolidator.consolidate(
        sources, start_date=start_date, end_date=end_date
    )
    matches = prepare_chronological_matches(matches)
    if matches.empty:
        raise RuntimeError("No resolved chronological matches remained after consolidation.")

    # Persist the canonical candidate store and requested full parquet. This is
    # separate from data/processed/matches.parquet used by the active model.
    consolidator.persist(
        matches,
        source_records,
        database_path=config.kaggle.candidate_matches_database_path,
        parquet_path=config.history.merged_matches_path,
    )
    matches = FilterEngine(config.filters).add_static_flags(matches)
    engineer = FeatureEngineer(config.features, config.filters)
    features = engineer.transform(matches, reset=True)

    leaked = set(FeatureEngineer.numeric_feature_columns()) & {
        "closing_odds_player1",
        "closing_odds_player2",
        "market_p1_open",
        "market_p1_close",
        "line_move_p1",
        "late_line_move_p1",
        "large_line_move",      # built from the closing line (data/odds.py): known only after the bet
        "player1_won",
        "score",
        "status",
    }
    if leaked:
        raise AssertionError(f"Outcome/odds leakage columns entered the feature list: {sorted(leaked)}")

    backtester = WalkForwardBacktester(config.model, config.backtest)
    predictions, metrics = backtester.run(features)
    calibration = calibration_curve_table(predictions)

    matches_path = Path(config.history.merged_matches_path)
    features_path = Path(config.history.chronological_features_path)
    predictions_path = output / "predictions.parquet"
    metrics_path = output / "metrics.csv"
    calibration_path = output / "calibration_curves.csv"
    unresolved_path = output / "identity_unresolved.parquet"
    candidate_model_path = output / "trained_system_candidate.joblib"

    _atomic_parquet(features_path, features)
    _atomic_parquet(predictions_path, predictions)
    metrics.to_csv(metrics_path, index=False)
    calibration.to_csv(calibration_path, index=False)
    _atomic_parquet(unresolved_path, unresolved)
    TennisPredictionSystem(config).fit(matches).save(candidate_model_path)

    production_predictions = _safe_frame(production_predictions_path)
    if production_predictions.empty:
        gate = ModelGateReport(
            status="INCONCLUSIVE",
            required_segments=list(_REQUIRED_GATE_SEGMENTS),
            passed_segments=[],
            failed_segments=[],
            inconclusive_segments=list(_REQUIRED_GATE_SEGMENTS),
            comparisons=[
                {
                    "segment": segment,
                    "status": "INCONCLUSIVE",
                    "reason": "production predictions unavailable",
                }
                for segment in _REQUIRED_GATE_SEGMENTS
            ],
        )
    else:
        gate = evaluate_paired_model_gate(
            predictions,
            production_predictions,
            minimum_segment_matches=config.history.minimum_gate_segment_matches,
            bootstrap_samples=config.history.gate_bootstrap_samples,
            confidence=config.history.gate_confidence,
            random_state=config.backtest.random_state,
        )

    (output / "promotion_gate.json").write_text(
        json.dumps(gate.to_dict(), indent=2), encoding="utf-8"
    )
    (output / "assembly_report.json").write_text(
        json.dumps(consolidation_report.to_dict(), indent=2), encoding="utf-8"
    )
    (output / "source_inventory.json").write_text(
        json.dumps(
            [
                {"source": source_name, "rows": len(frame)}
                for frame, source_name in sources
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    promoted = False
    if promote_if_passed:
        if gate.status != "PASS":
            raise RuntimeError(
                f"Candidate was not promoted because the paired gate is {gate.status}."
            )
        if not production_model_path:
            raise ValueError("production_model_path is required for promotion")
        production = Path(production_model_path)
        production.parent.mkdir(parents=True, exist_ok=True)
        if production.exists():
            backup = production.with_suffix(production.suffix + ".pre_candidate_backup")
            shutil.copy2(production, backup)
        shutil.copy2(candidate_model_path, production)
        promoted = True

    return CandidateBuildReport(
        matches_path=str(matches_path),
        features_path=str(features_path),
        predictions_path=str(predictions_path),
        metrics_path=str(metrics_path),
        calibration_path=str(calibration_path),
        candidate_model_path=str(candidate_model_path),
        unresolved_path=str(unresolved_path),
        assembly=consolidation_report.to_dict(),
        model_gate=gate.to_dict(),
        promoted=promoted,
    )
