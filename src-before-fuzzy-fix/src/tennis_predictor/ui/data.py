from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from tennis_predictor.config import AppConfig
from tennis_predictor.pipeline import TennisPredictionSystem



def load_app_config() -> AppConfig:
    path = Path(os.getenv("TENNIS_CONFIG_PATH", "config.yaml"))
    config = AppConfig.from_yaml(path) if path.exists() else AppConfig()
    paper_db = os.getenv("TENNIS_PAPER_DB")
    processed = os.getenv("TENNIS_PROCESSED_PATH")
    if paper_db:
        config.paper_trading.database_path = paper_db
    if processed:
        config.data.processed_path = processed
    config.ensure_directories()
    return config


def _mtime(path: str | Path) -> float:
    candidate = Path(path)
    return candidate.stat().st_mtime if candidate.exists() else -1.0


@st.cache_resource(show_spinner="Loading trained tennis model…")
def _load_system_cached(path: str, modified: float) -> TennisPredictionSystem:
    del modified
    return TennisPredictionSystem.load(path)


def load_system(path: str | Path) -> TennisPredictionSystem:
    return _load_system_cached(str(path), _mtime(path))


@st.cache_data(show_spinner="Loading backtest results…")
def _read_csv_cached(path: str, modified: float) -> pd.DataFrame:
    del modified
    return pd.read_csv(path)


@st.cache_data(show_spinner="Loading prediction history…")
def _read_parquet_cached(path: str, modified: float) -> pd.DataFrame:
    del modified
    return pd.read_parquet(path)


def load_metrics(path: str | Path) -> pd.DataFrame:
    return _read_csv_cached(str(path), _mtime(path))


def load_predictions(path: str | Path) -> pd.DataFrame:
    frame = _read_parquet_cached(str(path), _mtime(path)).copy()
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame


def clear_dashboard_cache() -> None:
    st.cache_resource.clear()
    st.cache_data.clear()


def player_options(system: TennisPredictionSystem, tour: str) -> list[tuple[str, str]]:
    resolver = system.name_resolver
    if resolver is None:
        return []
    identities = [identity for identity in resolver.identities if identity.tour == tour]
    identities.sort(key=lambda identity: identity.name.casefold())
    seen: set[str] = set()
    options: list[tuple[str, str]] = []
    for identity in identities:
        key = identity.name.casefold()
        if key in seen:
            continue
        seen.add(key)
        options.append((identity.name, identity.player_id))
    return options


def safe_load(path: Path, loader) -> tuple[Any | None, str | None]:
    if not path.exists():
        return None, f"Missing file: {path}"
    try:
        return loader(path), None
    except Exception as exc:  # UI boundary: surface a useful message instead of crashing the app
        return None, f"Could not load {path}: {exc}"
