from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class PlayerSampleQuality:
    name: str
    total_matches: int
    surface_matches: int
    last_match_date: str | None
    staleness_days: int | None


@dataclass(frozen=True, slots=True)
class MatchSampleQuality:
    status: str
    can_predict: bool
    can_bet: bool
    reason: str
    player1: PlayerSampleQuality
    player2: PlayerSampleQuality

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalized_calendar_day(value: object) -> pd.Timestamp:
    """Return a timezone-naive calendar day without shifting its displayed date.

    Polymarket times may already be converted into the board timezone, while
    stored player-state dates are commonly timezone-naive. Dropping timezone
    metadata preserves the caller's local calendar date and avoids converting
    a late-evening New York match into the following UTC day.
    """
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return pd.NaT

    timestamp = pd.Timestamp(timestamp)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)

    return timestamp.normalize()


def _player_quality(
    state: Any | None,
    *,
    name: str,
    surface: str,
    match_date: object,
) -> PlayerSampleQuality:
    total_matches = 0 if state is None else int(getattr(state, "total_matches", 0) or 0)
    surface_matches_map = {} if state is None else getattr(state, "surface_matches", {}) or {}
    surface_matches = int(surface_matches_map.get(surface, 0) or 0)

    last_date = None if state is None else getattr(state, "last_date", None)
    last_day = _normalized_calendar_day(last_date)
    target_day = _normalized_calendar_day(match_date)

    if pd.isna(last_day):
        last_text = None
        staleness_days = None
    else:
        last_text = last_day.date().isoformat()
        staleness_days = None if pd.isna(target_day) else max(int((target_day - last_day).days), 0)

    return PlayerSampleQuality(
        name=str(name),
        total_matches=total_matches,
        surface_matches=surface_matches,
        last_match_date=last_text,
        staleness_days=staleness_days,
    )


def evaluate_match_sample_quality(
    state1: Any | None,
    state2: Any | None,
    *,
    player1: str,
    player2: str,
    surface: str,
    match_date: object,
    absolute_min_matches: int = 8,
    recommended_matches: int = 20,
    recommended_surface_matches: int = 5,
    max_staleness_days: int = 120,
) -> MatchSampleQuality:
    """Classify whether a matchup has enough recorded history for betting use."""
    p1 = _player_quality(state1, name=player1, surface=surface, match_date=match_date)
    p2 = _player_quality(state2, name=player2, surface=surface, match_date=match_date)
    players = (p1, p2)

    insufficient = [
        f"{player.name} has only {player.total_matches} recorded matches"
        for player in players
        if player.total_matches < int(absolute_min_matches)
    ]
    if insufficient:
        return MatchSampleQuality(
            status="INSUFFICIENT",
            can_predict=False,
            can_bet=False,
            reason="; ".join(insufficient)
            + f". At least {int(absolute_min_matches)} are required to issue a prediction.",
            player1=p1,
            player2=p2,
        )

    limitations: list[str] = []
    for player in players:
        if player.total_matches < int(recommended_matches):
            limitations.append(
                f"{player.name}: {player.total_matches}/{int(recommended_matches)} total matches"
            )
        if player.surface_matches < int(recommended_surface_matches):
            limitations.append(
                f"{player.name}: {player.surface_matches}/{int(recommended_surface_matches)} {surface} matches"
            )

    recent_players = [
        player for player in players
        if player.staleness_days is not None
        and player.staleness_days <= int(max_staleness_days)
    ]
    if not recent_players:
        limitations.append(
            f"Neither player has a recorded match within {int(max_staleness_days)} days"
        )

    if limitations:
        return MatchSampleQuality(
            status="LIMITED",
            can_predict=True,
            can_bet=False,
            reason="; ".join(limitations)
            + ". Probability is research-only; do not produce a betting signal.",
            player1=p1,
            player2=p2,
        )

    return MatchSampleQuality(
        status="SUFFICIENT",
        can_predict=True,
        can_bet=True,
        reason=(
            f"Both players have at least {int(recommended_matches)} recorded matches, "
            f"at least {int(recommended_surface_matches)} on {surface}, and at least one recent player."
        ),
        player1=p1,
        player2=p2,
    )
