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
class LiveDataConfig:
    api_tennis_key_env: str = "API_TENNIS_KEY"
    timezone: str = "America/New_York"
    backfill_start: str = "2026-06-08"
    tournament_registry_path: str = "data/metadata/tournaments.csv"
    player_aliases_path: str = "data/metadata/player_aliases.csv"
    unresolved_fixtures_path: str = "data/live/unresolved_fixtures.csv"
    upcoming_days: int = 3
    replay_days: int = 7
    polymarket_refresh_seconds: int = 30
    polymarket_market_pages: int = 4
    polymarket_page_size: int = 100
    polymarket_max_markets_per_day: int = 60
    polymarket_book_batch_size: int = 50
    cold_start_min_matches: int = 8
    recommended_prediction_matches: int = 20
    recommended_surface_matches: int = 5
    max_player_staleness_days: int = 120


@dataclass(slots=True)
class KaggleDataConfig:
    enabled: bool = True
    cli_executable: str = "kaggle"
    raw_root: str = "data/raw/kaggle"
    normalized_root: str = "data/normalized/kaggle"
    candidate_matches_database_path: str = "data/candidate_matches.db"
    atp_daily_slug: str = "dissfya/atp-tennis-2000-2023daily-pull"
    wta_daily_slug: str = "dissfya/wta-tennis-2007-2023-daily-update"
    huge_tennis_slug: str = "guillemservera/tennis"
    command_timeout_seconds: int = 1200
    accept_schema_change: bool = False


@dataclass(slots=True)
class HistoricalDataConfig:
    start_date: str = "2023-01-01"
    raw_api_tennis_dir: str = "data/raw/api_tennis"
    normalized_api_tennis_path: str = "data/normalized/api_tennis_matches.parquet"
    unresolved_api_tennis_path: str = "data/normalized/api_tennis_unresolved.csv"
    tennis_data_directory: str | None = None
    tennis_data_snapshots_path: str = "data/normalized/tennis_data_odds.parquet"
    identity_database_path: str = "data/identity.db"
    merged_matches_path: str = "data/processed/matches_full.parquet"
    chronological_features_path: str = "data/processed/features_full.parquet"
    candidate_output_dir: str = "outputs/candidate"
    fuzzy_match_threshold: float = 96.0
    fuzzy_minimum_margin: float = 3.0
    max_retries: int = 4
    retry_backoff_seconds: float = 2.0
    sleep_between_requests: float = 0.25
    minimum_gate_segment_matches: int = 200
    gate_bootstrap_samples: int = 2000
    gate_confidence: float = 0.95



@dataclass(slots=True)
class PaperTradingConfig:
    database_path: str = "data/paper_trading.db"
    default_stake: float = 5.0
    min_edge: float = 0.03
    kelly_multiplier: float = 0.25
    max_kelly_fraction: float = 0.05


@dataclass(slots=True)
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    odds: OddsConfig = field(default_factory=OddsConfig)
    live_data: LiveDataConfig = field(default_factory=LiveDataConfig)
    history: HistoricalDataConfig = field(default_factory=HistoricalDataConfig)
    kaggle: KaggleDataConfig = field(default_factory=KaggleDataConfig)
    paper_trading: PaperTradingConfig = field(default_factory=PaperTradingConfig)

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
            live_data=LiveDataConfig(**raw.get("live_data", {})),
            history=HistoricalDataConfig(**raw.get("history", {})),
            kaggle=KaggleDataConfig(**raw.get("kaggle", {})),
            paper_trading=PaperTradingConfig(**raw.get("paper_trading", {})),
        )

    def ensure_directories(self) -> None:
        Path(self.data.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.data.processed_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.filters.context_dir).mkdir(parents=True, exist_ok=True)
        Path(self.live_data.tournament_registry_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.live_data.player_aliases_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.live_data.unresolved_fixtures_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.history.raw_api_tennis_dir).mkdir(parents=True, exist_ok=True)
        Path(self.history.normalized_api_tennis_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.history.tennis_data_snapshots_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.history.identity_database_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.history.merged_matches_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.history.candidate_output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.kaggle.raw_root).mkdir(parents=True, exist_ok=True)
        Path(self.kaggle.normalized_root).mkdir(parents=True, exist_ok=True)
        Path(self.kaggle.candidate_matches_database_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.paper_trading.database_path).parent.mkdir(parents=True, exist_ok=True)
