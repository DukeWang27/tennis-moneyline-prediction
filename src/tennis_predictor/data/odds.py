from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from rapidfuzz.fuzz import WRatio

from .names import normalize_name
from tennis_predictor.market.odds_math import devig_two_way


def load_normalized_snapshots(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "snapshot_time",
        "commence_time",
        "player1",
        "player2",
        "decimal_odds1",
        "decimal_odds2",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Odds snapshots are missing columns: {sorted(missing)}")
    for col in ("snapshot_time", "commence_time"):
        frame[col] = pd.to_datetime(frame[col], utc=True, errors="coerce").dt.tz_convert(None)
    frame["tour"] = frame.get("tour", "").astype("string").str.upper()
    frame["tournament"] = frame.get("tournament", "").astype("string").fillna("")
    frame["bookmaker"] = frame.get("bookmaker", "unknown").astype("string")
    frame["is_closing"] = frame.get("is_closing", False).astype(bool)
    return frame


@dataclass(slots=True)
class TheOddsApiClient:
    api_key: str | None = None
    base_url: str = "https://api.the-odds-api.com/v4"
    timeout_seconds: int = 45

    def _key(self, env_name: str = "ODDS_API_KEY") -> str:
        key = self.api_key or os.getenv(env_name)
        if not key:
            raise RuntimeError(f"Set {env_name} or pass api_key to use The Odds API.")
        return key

    def fetch_sports(self, *, all_sports: bool = False, env_name: str = "ODDS_API_KEY") -> list[dict[str, Any]]:
        response = requests.get(
            f"{self.base_url}/sports",
            params={"apiKey": self._key(env_name), "all": str(all_sports).lower()},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, list) else []

    def fetch_upcoming_odds(
        self,
        sport_key: str,
        *,
        regions: str = "us,uk,eu",
        bookmakers: str | None = None,
        env_name: str = "ODDS_API_KEY",
    ) -> pd.DataFrame:
        params = {
            "apiKey": self._key(env_name),
            "markets": "h2h",
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        }
        # The Odds API treats `bookmakers` as an alternative to `regions`.
        # Supplying only the requested bookmaker avoids spending quota on books
        # that the dashboard will never display.
        if bookmakers:
            params["bookmakers"] = bookmakers
        else:
            params["regions"] = regions
        response = requests.get(
            f"{self.base_url}/sports/{sport_key}/odds",
            params=params,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return self.normalize_upcoming_payload(response.json(), sport_key=sport_key)

    def fetch_all_tennis_odds(
        self,
        *,
        regions: str = "us,uk,eu",
        bookmakers: str | None = None,
        env_name: str = "ODDS_API_KEY",
    ) -> pd.DataFrame:
        sports = self.fetch_sports(env_name=env_name)
        tennis_keys = [
            str(item.get("key"))
            for item in sports
            if "tennis" in str(item.get("group", "")).casefold()
            or str(item.get("key", "")).startswith("tennis_")
        ]
        frames = []
        for sport_key in tennis_keys:
            frame = self.fetch_upcoming_odds(
                sport_key, regions=regions, bookmakers=bookmakers, env_name=env_name
            )
            if not frame.empty:
                frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    @staticmethod
    def normalize_upcoming_payload(payload: Any, *, sport_key: str) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        events = payload if isinstance(payload, list) else []
        snapshot_time = pd.Timestamp.now(tz="UTC").tz_convert(None)
        for event in events:
            home = event.get("home_team")
            away = event.get("away_team")
            if not home or not away:
                continue
            for book in event.get("bookmakers", []):
                for market in book.get("markets", []):
                    if market.get("key") != "h2h":
                        continue
                    prices = {outcome.get("name"): outcome.get("price") for outcome in market.get("outcomes", [])}
                    if home not in prices or away not in prices:
                        continue
                    rows.append(
                        {
                            "provider_event_id": str(event.get("id")),
                            "snapshot_time": snapshot_time,
                            "commence_time": pd.to_datetime(event.get("commence_time"), utc=True, errors="coerce").tz_convert(None),
                            "tour": "",
                            "tournament": event.get("sport_title", sport_key),
                            "sport_key": sport_key,
                            "player1": str(home),
                            "player2": str(away),
                            "bookmaker": str(book.get("title") or book.get("key") or "unknown"),
                            "bookmaker_key": str(book.get("key") or "unknown"),
                            "decimal_odds1": float(prices[home]),
                            "decimal_odds2": float(prices[away]),
                            "source": "the_odds_api",
                            "is_closing": False,
                        }
                    )
        return pd.DataFrame(rows)

    def fetch_historical_snapshot(
        self,
        sport_key: str,
        at: str | pd.Timestamp,
        *,
        regions: str = "us,uk,eu",
        bookmakers: str | None = None,
        env_name: str = "ODDS_API_KEY",
    ) -> pd.DataFrame:
        """Fetch one paid historical snapshot and normalize H2H bookmaker rows."""
        at_ts = pd.Timestamp(at, tz="UTC") if pd.Timestamp(at).tzinfo is None else pd.Timestamp(at)
        params = {
            "apiKey": self._key(env_name),
            "regions": regions,
            "markets": "h2h",
            "oddsFormat": "decimal",
            "dateFormat": "iso",
            "date": at_ts.isoformat().replace("+00:00", "Z"),
        }
        if bookmakers:
            params["bookmakers"] = bookmakers
        response = requests.get(
            f"{self.base_url}/historical/sports/{sport_key}/odds",
            params=params,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return self.normalize_payload(payload, at_ts)

    @staticmethod
    def normalize_payload(payload: dict[str, Any], snapshot_time: pd.Timestamp) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for event in payload.get("data", []):
            home = event.get("home_team")
            away = event.get("away_team")
            for book in event.get("bookmakers", []):
                for market in book.get("markets", []):
                    if market.get("key") != "h2h":
                        continue
                    prices = {x.get("name"): x.get("price") for x in market.get("outcomes", [])}
                    if home not in prices or away not in prices:
                        continue
                    rows.append(
                        {
                            "provider_event_id": event.get("id"),
                            "snapshot_time": snapshot_time,
                            "commence_time": pd.to_datetime(event.get("commence_time"), utc=True).tz_convert(None),
                            "tour": "",
                            "tournament": event.get("sport_title", ""),
                            "player1": home,
                            "player2": away,
                            "bookmaker": book.get("key", "unknown"),
                            "decimal_odds1": float(prices[home]),
                            "decimal_odds2": float(prices[away]),
                            "is_closing": False,
                        }
                    )
        return pd.DataFrame(rows)


class OddsAligner:
    """Align provider snapshots to matches with an auditable similarity score."""

    def __init__(self, name_threshold: float = 88.0, total_threshold: float = 82.0):
        self.name_threshold = name_threshold
        self.total_threshold = total_threshold

    @staticmethod
    def _pair_score(a1: str, a2: str, b1: str, b2: str) -> tuple[float, bool]:
        direct = (WRatio(a1, b1) + WRatio(a2, b2)) / 2
        reverse = (WRatio(a1, b2) + WRatio(a2, b1)) / 2
        return (direct, False) if direct >= reverse else (reverse, True)

    def align(self, matches: pd.DataFrame, snapshots: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        matches = matches.copy()
        snapshots = snapshots.copy()
        match_names = matches.assign(
            _p1=matches["player1"].map(normalize_name),
            _p2=matches["player2"].map(normalize_name),
            _t=matches["tournament"].map(normalize_name),
        )
        aligned_rows: list[dict[str, Any]] = []
        for row in snapshots.itertuples(index=False):
            event_date = pd.Timestamp(row.commence_time).normalize()
            candidates = match_names[
                (match_names["date"].dt.normalize().between(event_date - pd.Timedelta(days=2), event_date + pd.Timedelta(days=2)))
            ]
            row_tour = str(getattr(row, "tour", "") or "").upper()
            if row_tour in {"ATP", "WTA"}:
                candidates = candidates[candidates["tour"] == row_tour]
            best: tuple[float, pd.Series, bool] | None = None
            o1, o2 = normalize_name(row.player1), normalize_name(row.player2)
            ot = normalize_name(getattr(row, "tournament", ""))
            for _, candidate in candidates.iterrows():
                name_score, reversed_order = self._pair_score(o1, o2, candidate["_p1"], candidate["_p2"])
                if name_score < self.name_threshold:
                    continue
                tournament_score = WRatio(ot, candidate._t) if ot and candidate._t else 70.0
                date_days = abs((event_date - candidate["date"].normalize()).days)
                date_score = max(0.0, 100.0 - date_days * 20.0)
                total = 0.70 * name_score + 0.20 * tournament_score + 0.10 * date_score
                if best is None or total > best[0]:
                    best = (total, candidate, reversed_order)
            if best is None or best[0] < self.total_threshold:
                continue
            score, candidate, reversed_order = best
            data = row._asdict()
            if reversed_order:
                data["decimal_odds1"], data["decimal_odds2"] = data["decimal_odds2"], data["decimal_odds1"]
            data.update(
                {
                    "match_id": candidate["match_id"],
                    "alignment_score": score,
                    "provider_order_reversed": reversed_order,
                }
            )
            aligned_rows.append(data)

        aligned = pd.DataFrame(aligned_rows)
        if aligned.empty:
            return matches, aligned
        summary = self.summarize(aligned)
        output = matches.merge(summary, on="match_id", how="left")

        # Exact provider timestamps supersede Sackmann's estimated round date.
        exact_dates = aligned.groupby("match_id")["commence_time"].median().dt.normalize()
        mask = output["match_id"].isin(exact_dates.index)
        output.loc[mask, "date"] = output.loc[mask, "match_id"].map(exact_dates)
        output.loc[mask, "date_is_estimated"] = False
        return output, aligned

    @staticmethod
    def summarize(aligned: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for match_id, group in aligned.groupby("match_id", sort=False):
            group = group.sort_values("snapshot_time").copy()
            commence = pd.to_datetime(group["commence_time"], errors="coerce").median()
            if pd.notna(commence):
                group = group[group["snapshot_time"] <= commence]
            snapshot = (
                group.groupby("snapshot_time", as_index=False)
                .agg(
                    decimal_odds1=("decimal_odds1", "median"),
                    decimal_odds2=("decimal_odds2", "median"),
                    alignment_score=("alignment_score", "max"),
                )
                .sort_values("snapshot_time")
            )
            if snapshot.empty:
                continue
            opening = snapshot.iloc[0]
            closing = snapshot.iloc[-1]
            late_open = opening
            if pd.notna(commence):
                late = snapshot[snapshot["snapshot_time"] >= commence - pd.Timedelta(hours=6)]
                if not late.empty:
                    late_open = late.iloc[0]
            open_p1, open_p2 = devig_two_way(opening.decimal_odds1, opening.decimal_odds2, fmt="decimal")
            late_p1, _ = devig_two_way(late_open.decimal_odds1, late_open.decimal_odds2, fmt="decimal")
            close_p1, close_p2 = devig_two_way(closing.decimal_odds1, closing.decimal_odds2, fmt="decimal")
            rows.append(
                {
                    "match_id": match_id,
                    "market_decimal1_open": opening.decimal_odds1,
                    "market_decimal2_open": opening.decimal_odds2,
                    "market_decimal1_close": closing.decimal_odds1,
                    "market_decimal2_close": closing.decimal_odds2,
                    "market_p1_open": open_p1,
                    "market_p2_open": open_p2,
                    "market_p1_close": close_p1,
                    "market_p2_close": close_p2,
                    "line_move_p1": close_p1 - open_p1,
                    "late_line_move_p1": close_p1 - late_p1,
                    "large_line_move": abs(close_p1 - late_p1) > 0.10,
                    "odds_snapshot_count": len(snapshot),
                    "odds_alignment_score": snapshot.alignment_score.max(),
                }
            )
        return pd.DataFrame(rows)
