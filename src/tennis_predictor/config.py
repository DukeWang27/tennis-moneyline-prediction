from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DataConfig:
    start_year: int = 2000
    end_year: int = 2026
    cache_dir: str = "data/raw"
    processed_path: str = "data/processed/matches.parquet"
    include_lower_levels: bool = True
    request_timeout_seconds: int = 45
    tier_overrides_path: str | None = None
    exact_dates_path: str | None = None


@dataclass(slots=True)
class FilterConfig:
    context_dir: str = "data/context"
    fatigue_matches: int = 3
    fatigue_window_days: int = 4
    pre_slam_window_days: int = 14
    low_tier_before_slam: list[str] = field(
        default_factory=lambda: ["250", "Challenger", "ITF"]
    )


@dataclass(slots=True)
class FeatureConfig:
    initial_elo: float = 1500.0
    elo_k: float = 32.0
    surface_prior_matches: float = 20.0
    retirement_elo_weight: float = 0.25
    walkover_elo_weight: float = 0.0
    recent_form_half_life_days: float = 30.0
    h2h_min_matches: int = 3
    rest_days_cap: int = 60


@dataclass(slots=True)
class ModelConfig:
    model_type: str = "boosted"
    calibration: str = "isotonic"
    random_state: int = 27
    lightgbm_params: dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 350,
            "learning_rate": 0.035,
            "num_leaves": 24,
            "max_depth": 6,
            "min_child_samples": 80,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.25,
            "reg_lambda": 1.0,
            "verbosity": -1,
        }
    )


@dataclass(slots=True)
class BacktestConfig:
    min_train_years: int = 3
    test_months: int = 3
    calibration_days: int = 180
    min_calibration_rows: int = 500
    edge_threshold: float = 0.02
    kelly_fraction: float = 0.25
    max_kelly_fraction: float = 0.05
    bootstrap_samples: int = 1000
    random_state: int = 27


@dataclass(slots=True)
class OddsConfig:
    normalized_snapshots_path: str | None = None
    api_key_env: str = "ODDS_API_KEY"
    regions: str = "us,uk,eu"
    bookmakers: str | None = None


@dataclass(slots=True)
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    odds: OddsConfig = field(default_factory=OddsConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        return cls(
            data=DataConfig(**raw.get("data", {})),
            filters=FilterConfig(**raw.get("filters", {})),
            features=FeatureConfig(**raw.get("features", {})),
            model=ModelConfig(**raw.get("model", {})),
            backtest=BacktestConfig(**raw.get("backtest", {})),
            odds=OddsConfig(**raw.get("odds", {})),
        )

    def ensure_directories(self) -> None:
        Path(self.data.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.data.processed_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.filters.context_dir).mkdir(parents=True, exist_ok=True)
