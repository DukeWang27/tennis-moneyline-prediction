"""Extra pre-match features from the raw Sackmann columns the standard loader drops.

Physical and style: height, handedness and the lefty-vs-righty matchup.
How they play: recency-weighted serve and return points won, ace and double-fault rates, break points saved/converted.
Fatigue: minutes on court in the last 3 and 7 days, matches in the last 14 days, a deep run the previous week.
Travel and home crowd: playing in their home country and changing country on short rest. A tournament's country is
inferred from the nationality most of its wildcards share (wildcards go mostly to home players), because the files
have no venue column.
Status: qualifier, wildcard, lucky loser, seeded, ranking-points ratio.

Every value is computed from matches strictly before the one it describes. Matches that share a date are all
snapshotted before any of them updates the state, so same-day results cannot leak into each other.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .chronology import enforce_round_order

RAW_COLUMNS = [
    "tourney_id", "match_num", "winner_id", "loser_id", "winner_ht", "loser_ht", "winner_hand", "loser_hand",
    "winner_ioc", "loser_ioc", "winner_entry", "loser_entry", "winner_seed", "loser_seed", "winner_rank_points",
    "loser_rank_points", "minutes", "w_ace", "w_df", "w_svpt", "w_1stWon", "w_2ndWon", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stWon", "l_2ndWon", "l_bpSaved", "l_bpFaced",
]

# Prior points: a player with no history is assumed tour-average, and real data takes over as it accumulates.
PRIOR_WEIGHT = 300.0  # serve/return points (about four matches)
PRIOR_BP_WEIGHT = 20.0
TOUR_SERVE_WIN = {"ATP": 0.63, "WTA": 0.56}
TOUR_ACE_RATE = {"ATP": 0.075, "WTA": 0.04}
TOUR_DF_RATE = {"ATP": 0.035, "WTA": 0.045}
TOUR_BP_SAVED = {"ATP": 0.60, "WTA": 0.53}
DECAY_PER_MATCH = 0.98  # older matches count less: weight halves after about 35 matches


def load_raw_match_details(raw_dir: str | Path) -> pd.DataFrame:
    """Read the yearly Sackmann CSVs and return one row per match_id (same id rule as SackmannLoader)."""
    frames = []
    for tour in ("atp", "wta"):
        for path in sorted((Path(raw_dir) / tour).glob(f"{tour}_matches_*.csv")):
            raw = pd.read_csv(path, usecols=lambda c: c in RAW_COLUMNS, low_memory=False)
            match_num = pd.to_numeric(raw["match_num"], errors="coerce").fillna(-1).astype(int)
            raw["match_id"] = tour.upper() + "_" + raw["tourney_id"].astype("string").fillna("unknown") + "_" + match_num.astype(str)
            raw["tour"] = tour.upper()
            frames.append(raw)
    details = pd.concat(frames, ignore_index=True)
    return details.drop_duplicates("match_id", keep="last").set_index("match_id")


def infer_tournament_countries(details: pd.DataFrame, min_wildcards: int = 2) -> dict[str, str]:
    """tourney_id -> IOC code, when at least ``min_wildcards`` wildcard entrants share a nationality and they are the
    clear majority of the event's wildcards. Events without that signal are left out (unknown, not guessed)."""
    rows = []
    for side in ("winner", "loser"):
        wc = details[details[f"{side}_entry"].astype("string").str.upper().eq("WC")]
        rows.append(pd.DataFrame({"tourney_id": wc["tourney_id"], "ioc": wc[f"{side}_ioc"], "player": wc[f"{side}_id"]}))
    wildcards = pd.concat(rows).dropna().drop_duplicates(["tourney_id", "player"])
    countries = {}
    for tourney_id, group in wildcards.groupby("tourney_id"):
        counts = group["ioc"].value_counts()
        if counts.iloc[0] >= min_wildcards and counts.iloc[0] / counts.sum() >= 0.6:
            countries[str(tourney_id)] = str(counts.index[0])
    return countries


@dataclass(slots=True)
class _Player:
    serve_won: float = 0.0
    serve_pts: float = 0.0
    return_won: float = 0.0
    return_pts: float = 0.0
    aces: float = 0.0
    dfs: float = 0.0
    bp_saved: float = 0.0
    bp_faced: float = 0.0
    bp_won: float = 0.0
    bp_chances: float = 0.0
    stat_matches: int = 0
    height: float = float("nan")
    hand: str = ""
    ioc: str = ""
    recent: deque = field(default_factory=lambda: deque(maxlen=20))  # (date, minutes, tourney_id)
    last_country: str = ""
    last_tourney: str = ""
    last_tourney_wins: int = 0
    last_date: pd.Timestamp | None = None


def _num(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _ok(value: float) -> bool:
    """True unless the value is NaN."""
    return not math.isnan(value)


def _rate(won: float, total: float, prior: float, weight: float) -> float:
    return (won + prior * weight) / (total + weight)


class ExtendedFeatureBuilder:
    def __init__(self, details: pd.DataFrame, countries: dict[str, str]):
        self.details = details
        self.countries = countries
        self.players: dict[str, _Player] = defaultdict(_Player)

    def _snapshot(self, pid: str, tour: str, date: pd.Timestamp, country: str) -> dict:
        p = self.players[pid]
        serve_prior = TOUR_SERVE_WIN.get(tour, 0.6)
        minutes3 = sum(m for d, m, _ in p.recent if (date - d).days <= 3 and _ok(m))
        minutes7 = sum(m for d, m, _ in p.recent if (date - d).days <= 7 and _ok(m))
        matches14 = sum(1 for d, _, _ in p.recent if (date - d).days <= 14)
        days_since = (date - p.last_date).days if p.last_date is not None else np.nan
        return {
            "serve_won": _rate(p.serve_won, p.serve_pts, serve_prior, PRIOR_WEIGHT),
            "return_won": _rate(p.return_won, p.return_pts, 1 - serve_prior, PRIOR_WEIGHT),
            "ace_rate": _rate(p.aces, p.serve_pts, TOUR_ACE_RATE.get(tour, 0.06), PRIOR_WEIGHT),
            "df_rate": _rate(p.dfs, p.serve_pts, TOUR_DF_RATE.get(tour, 0.04), PRIOR_WEIGHT),
            "bp_saved": _rate(p.bp_saved, p.bp_faced, TOUR_BP_SAVED.get(tour, 0.57), PRIOR_BP_WEIGHT),
            "bp_converted": _rate(p.bp_won, p.bp_chances, 1 - TOUR_BP_SAVED.get(tour, 0.57), PRIOR_BP_WEIGHT),
            "stat_matches": p.stat_matches,
            "minutes_3d": minutes3,
            "minutes_7d": minutes7,
            "matches_14d": matches14,
            "deep_run_last_event": float(p.last_tourney_wins >= 3 and _ok(days_since) and days_since <= 10),
            "changed_country_short_rest": float(
                bool(country) and bool(p.last_country) and country != p.last_country and _ok(days_since) and days_since <= 5
            ),
            "home": float(bool(country) and p.ioc == country),
            "height": p.height,
            "lefty": float(p.hand == "L"),
        }

    def _update(self, pid: str, opp: str, won: bool, row, prefix: str, opp_prefix: str, date: pd.Timestamp,
                tourney: str, country: str, minutes: float) -> None:
        p = self.players[pid]
        height = _num(getattr(row, f"{'winner' if won else 'loser'}_ht"))
        if _ok(height):
            p.height = height
        hand = getattr(row, f"{'winner' if won else 'loser'}_hand")
        if isinstance(hand, str) and hand in "LR":
            p.hand = hand
        ioc = getattr(row, f"{'winner' if won else 'loser'}_ioc")
        if isinstance(ioc, str):
            p.ioc = ioc
        svpt = _num(getattr(row, f"{prefix}_svpt"))
        opp_svpt = _num(getattr(row, f"{opp_prefix}_svpt"))
        if _ok(svpt) and svpt > 0 and _ok(opp_svpt) and opp_svpt > 0:
            serve_won = _num(getattr(row, f"{prefix}_1stWon")) + _num(getattr(row, f"{prefix}_2ndWon"))
            opp_serve_won = _num(getattr(row, f"{opp_prefix}_1stWon")) + _num(getattr(row, f"{opp_prefix}_2ndWon"))
            if _ok(serve_won) and _ok(opp_serve_won):
                d = DECAY_PER_MATCH
                p.serve_won = p.serve_won * d + serve_won
                p.serve_pts = p.serve_pts * d + svpt
                p.return_won = p.return_won * d + (opp_svpt - opp_serve_won)
                p.return_pts = p.return_pts * d + opp_svpt
                p.aces = p.aces * d + np.nan_to_num(_num(getattr(row, f"{prefix}_ace")))
                p.dfs = p.dfs * d + np.nan_to_num(_num(getattr(row, f"{prefix}_df")))
                faced = np.nan_to_num(_num(getattr(row, f"{prefix}_bpFaced")))
                p.bp_saved = p.bp_saved * d + np.nan_to_num(_num(getattr(row, f"{prefix}_bpSaved")))
                p.bp_faced = p.bp_faced * d + faced
                opp_faced = np.nan_to_num(_num(getattr(row, f"{opp_prefix}_bpFaced")))
                opp_saved = np.nan_to_num(_num(getattr(row, f"{opp_prefix}_bpSaved")))
                p.bp_won = p.bp_won * d + (opp_faced - opp_saved)
                p.bp_chances = p.bp_chances * d + opp_faced
                p.stat_matches += 1
        p.recent.append((date, minutes, tourney))
        if tourney != p.last_tourney:
            p.last_tourney_wins = 0
        p.last_tourney = tourney
        p.last_tourney_wins += int(won)
        if country:
            p.last_country = country
        p.last_date = date

    def build(self, matches: pd.DataFrame) -> pd.DataFrame:
        """``matches`` needs match_id, date (or chronology_ts), tour, player1_id, player2_id, player1_won."""
        order = matches.assign(_ts=pd.to_datetime(matches.get("chronology_ts", matches["date"])))
        if "round" in order:
            order = enforce_round_order(order)
            order["_ts"] = order["chronology_ts"]
        order = order.sort_values(["_ts", "match_id"], kind="mergesort")
        details = self.details.reindex(order["match_id"])
        out: dict[str, dict] = {}
        empty = pd.Series(dtype=object)
        for ts, group in order.groupby("_ts", sort=True):
            pending = []
            for m in group.itertuples(index=False):
                row = details.loc[m.match_id] if m.match_id in details.index else empty
                tourney = str(row.get("tourney_id", "")) if len(row) else ""
                country = self.countries.get(tourney, "")
                ts_date = pd.Timestamp(ts)
                s1 = self._snapshot(str(m.player1_id), str(m.tour), ts_date, country)
                s2 = self._snapshot(str(m.player2_id), str(m.tour), ts_date, country)
                feats = {f"x_{k}1": v for k, v in s1.items()} | {f"x_{k}2": v for k, v in s2.items()}
                for k in ("serve_won", "return_won", "ace_rate", "df_rate", "bp_saved", "bp_converted", "minutes_3d",
                          "minutes_7d", "matches_14d", "deep_run_last_event", "changed_country_short_rest", "home", "height"):
                    feats[f"x_{k}_diff"] = s1[k] - s2[k]
                # Expected serve advantage: my serve against their return, minus theirs against mine.
                feats["x_serve_return_edge"] = (s1["serve_won"] - s2["return_won"]) - (s2["serve_won"] - s1["return_won"])
                feats["x_lefty_vs_righty"] = float(s1["lefty"] != s2["lefty"])
                if len(row):
                    p1_won = int(m.player1_won) == 1
                    w, l = ("1", "2") if p1_won else ("2", "1")
                    for side, src in ((w, "winner"), (l, "loser")):
                        entry = str(row.get(f"{src}_entry") or "").upper()
                        feats[f"x_qualifier{side}"] = float(entry == "Q")
                        feats[f"x_wildcard{side}"] = float(entry == "WC")
                        feats[f"x_lucky_loser{side}"] = float(entry == "LL")
                        feats[f"x_seeded{side}"] = float(_num(row.get(f"{src}_seed")) == _num(row.get(f"{src}_seed")))
                        feats[f"x_rank_points{side}"] = _num(row.get(f"{src}_rank_points"))
                    pts1, pts2 = feats["x_rank_points1"], feats["x_rank_points2"]
                    feats["x_log_points_ratio"] = (
                        math.log((pts1 + 1) / (pts2 + 1)) if _ok(pts1) and _ok(pts2) else np.nan
                    )
                    pending.append((m, row, p1_won, tourney, country, _num(row.get("minutes"))))
                out[m.match_id] = feats
            for m, row, p1_won, tourney, country, minutes in pending:
                if bool(getattr(m, "walkover", False)):
                    continue
                r = _RowView(row)
                winner, loser = (str(m.player1_id), str(m.player2_id)) if p1_won else (str(m.player2_id), str(m.player1_id))
                self._update(winner, loser, True, r, "w", "l", pd.Timestamp(ts), tourney, country, minutes)
                self._update(loser, winner, False, r, "l", "w", pd.Timestamp(ts), tourney, country, minutes)
        return pd.DataFrame.from_dict(out, orient="index").rename_axis("match_id").reset_index()


class _RowView:
    """Attribute access over a pandas Series, returning NaN for missing columns."""

    __slots__ = ("_row",)

    def __init__(self, row: pd.Series):
        self._row = row

    def __getattr__(self, name: str):
        return self._row.get(name, np.nan)


EXTENDED_DIFF_FEATURES = [
    "x_serve_won_diff", "x_return_won_diff", "x_serve_return_edge", "x_ace_rate_diff", "x_df_rate_diff",
    "x_bp_saved_diff", "x_bp_converted_diff", "x_minutes_3d_diff", "x_minutes_7d_diff", "x_matches_14d_diff",
    "x_deep_run_last_event_diff", "x_changed_country_short_rest_diff", "x_home_diff", "x_height_diff",
    "x_lefty1", "x_lefty2", "x_lefty_vs_righty", "x_qualifier1", "x_qualifier2", "x_wildcard1", "x_wildcard2",
    "x_lucky_loser1", "x_lucky_loser2", "x_seeded1", "x_seeded2", "x_log_points_ratio",
]

