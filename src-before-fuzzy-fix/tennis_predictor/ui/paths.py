from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DashboardPaths:
    model: Path
    metrics: Path
    predictions: Path

    @classmethod
    def defaults(cls) -> "DashboardPaths":
        output_dir = Path(os.getenv("TENNIS_OUTPUT_DIR", "outputs/backtest"))
        deploy_dir = Path(os.getenv("TENNIS_DASHBOARD_ARTIFACTS", "dashboard_artifacts"))

        def choose(filename: str) -> Path:
            explicit = os.getenv(
                {
                    "trained_system.joblib": "TENNIS_MODEL_PATH",
                    "metrics.csv": "TENNIS_METRICS_PATH",
                    "predictions.parquet": "TENNIS_PREDICTIONS_PATH",
                }[filename]
            )
            if explicit:
                return Path(explicit)
            local = output_dir / filename
            return local if local.exists() else deploy_dir / filename

        return cls(
            model=choose("trained_system.joblib"),
            metrics=choose("metrics.csv"),
            predictions=choose("predictions.parquet"),
        )

    def with_overrides(self, *, model: str, metrics: str, predictions: str) -> "DashboardPaths":
        return DashboardPaths(Path(model), Path(metrics), Path(predictions))

    def status(self) -> dict[str, bool]:
        return {
            "Model": self.model.exists(),
            "Metrics": self.metrics.exists(),
            "Predictions": self.predictions.exists(),
        }
