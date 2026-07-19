from __future__ import annotations

import numpy as np
import pandas as pd


HISTORICAL_REPLAY_COLUMNS = [
    "match_id",
    "date",
    "tour",
    "tier",
    "surface",
    "tournament",
    "round",
    "player1",
    "player2",
    "actual",
    "boosted_probability",
    "elo_probability",
    "fold",
    "retirement",
    "walkover",
    "eligible_with_filters",
    "any_filter_flag",
]


def prepare_historical_replay(predictions: pd.DataFrame) -> pd.DataFrame:
    """Build one leakage-safe row per match from saved walk-forward output.

    The current fitted model is never called. Probabilities come only from the
    chronological out-of-sample predictions already written by the backtester.
    """
    if predictions is None or predictions.empty:
        return pd.DataFrame(columns=HISTORICAL_REPLAY_COLUMNS)

    required = {
        "match_id",
        "date",
        "model",
        "probability",
        "actual",
        "player1",
        "player2",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(
            "Historical replay requires walk-forward prediction columns: "
            f"{sorted(missing)}"
        )

    frame = predictions.copy()
    frame["match_id"] = frame["match_id"].astype(str)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["probability"] = pd.to_numeric(frame["probability"], errors="coerce")
    frame["actual"] = pd.to_numeric(frame["actual"], errors="coerce")
    frame = frame[
        frame["model"].isin(["boosted", "elo_baseline"])
        & frame["date"].notna()
        & frame["probability"].notna()
    ].copy()
    if frame.empty:
        return pd.DataFrame(columns=HISTORICAL_REPLAY_COLUMNS)

    metadata_candidates = [
        "match_id",
        "date",
        "tour",
        "tier",
        "surface",
        "tournament",
        "round",
        "player1",
        "player2",
        "actual",
        "fold",
        "retirement",
        "walkover",
        "eligible_with_filters",
        "any_filter_flag",
    ]
    metadata_columns = [column for column in metadata_candidates if column in frame.columns]
    metadata = (
        frame.sort_values(["match_id", "model"])
        .drop_duplicates("match_id", keep="first")[metadata_columns]
        .copy()
    )

    probabilities = (
        frame.pivot_table(
            index="match_id",
            columns="model",
            values="probability",
            aggfunc="first",
        )
        .rename(
            columns={
                "boosted": "boosted_probability",
                "elo_baseline": "elo_probability",
            }
        )
        .reset_index()
    )
    replay = metadata.merge(probabilities, on="match_id", how="left")
    if "boosted_probability" not in replay.columns:
        replay["boosted_probability"] = np.nan
    if "elo_probability" not in replay.columns:
        replay["elo_probability"] = np.nan
    replay = replay[replay["boosted_probability"].notna()].copy()

    for column in HISTORICAL_REPLAY_COLUMNS:
        if column not in replay.columns:
            replay[column] = np.nan
    replay = replay[HISTORICAL_REPLAY_COLUMNS]
    replay = replay.sort_values(["date", "match_id"]).reset_index(drop=True)
    return replay


UNRESOLVED_DISPLAY_COLUMNS = [
    "event_date",
    "event_time",
    "matchup",
    "tournament",
    "tour",
    "tier",
    "status",
    "reason",
]


def clean_unresolved_fixtures(
    unresolved: pd.DataFrame,
    *,
    limit: int | None = 250,
) -> pd.DataFrame:
    """Return a compact unresolved-fixture table safe for dashboard display."""
    if unresolved is None or unresolved.empty:
        return pd.DataFrame(columns=UNRESOLVED_DISPLAY_COLUMNS)

    frame = unresolved.copy()

    def first_available(*names: str, default: object = "") -> pd.Series:
        for name in names:
            if name in frame.columns:
                return frame[name]
        return pd.Series([default] * len(frame), index=frame.index)

    first_player = first_available("event_first_player", "player1").fillna("").astype(str).str.strip()
    second_player = first_available("event_second_player", "player2").fillna("").astype(str).str.strip()
    matchup = first_player.str.cat(second_player, sep=" vs ").str.strip()
    matchup = matchup.mask(matchup.eq("vs"), "Players unavailable")

    display = pd.DataFrame(
        {
            "event_date": pd.to_datetime(
                first_available("event_date", "date"), errors="coerce"
            ),
            "event_time": first_available("event_time").fillna("").astype(str),
            "matchup": matchup,
            "tournament": first_available("tournament_name", "tournament", default="Unknown")
            .fillna("Unknown")
            .astype(str),
            "tour": first_available("tour", default="Unknown").fillna("Unknown").astype(str),
            "tier": first_available("tier", default="Unknown").fillna("Unknown").astype(str),
            "status": first_available("event_status", default="Unknown").fillna("Unknown").astype(str),
            "reason": first_available("unresolved_reason", default="unresolved")
            .fillna("unresolved")
            .astype(str),
        }
    )
    display = display.drop_duplicates().sort_values(
        ["reason", "event_date", "event_time", "matchup"],
        na_position="last",
    )
    if limit is not None:
        display = display.head(max(int(limit), 0))
    return display.reset_index(drop=True)
