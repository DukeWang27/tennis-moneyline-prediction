"""Keep each tournament's rounds in order before features are built.

Exact match times from an odds feed replace Sackmann's estimated round dates only for the matches that have odds. The
other matches in the same event keep estimates, so a player's next-round match could be sorted before the match it
depends on, and its existence gives the result away ("this player is still in the draw tomorrow"). In the 2021-2024
Pinnacle run that happened to 5.6% of priced matches, and same-day ties were broken by Sackmann's match number, which
does not always follow round order.

``enforce_round_order`` makes ``chronology_ts`` non-decreasing through each event's rounds and adds ``round_order``
as the tie-break for matches that share a timestamp.
"""

from __future__ import annotations

import pandas as pd

ROUND_ORDER = {
    "Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3,
    "ER": 4, "RR": 5, "R128": 6, "R64": 7, "R32": 8, "R16": 9, "QF": 10, "SF": 11, "BR": 12, "F": 13,
}


def event_key(frame: pd.DataFrame) -> pd.Series:
    """Sackmann ids look like ``ATP_2023-0339_271``: everything before the last underscore is the event."""
    ids = frame["match_id"].astype("string")
    from_id = ids.str.rsplit("_", n=1).str[0]
    fallback = (
        frame.get("tour", pd.Series("", index=frame.index)).astype("string").fillna("")
        + "|" + frame.get("tournament", pd.Series("", index=frame.index)).astype("string").fillna("")
        + "|" + pd.to_datetime(frame.get("event_date", frame["date"]), errors="coerce").astype("string").fillna("")
    )
    return from_id.where(ids.str.contains("_", regex=False).fillna(False), fallback)


def enforce_round_order(matches: pd.DataFrame) -> pd.DataFrame:
    output = matches.copy()
    if "chronology_ts" not in output:
        output["chronology_ts"] = pd.to_datetime(output["date"], errors="coerce")
    output["chronology_ts"] = pd.to_datetime(output["chronology_ts"], errors="coerce")
    rounds = output.get("round", pd.Series("", index=output.index)).astype("string").str.upper().str.strip()
    output["round_order"] = rounds.map(ROUND_ORDER).astype("float").fillna(-1.0)
    known = output["round_order"] >= 0
    if not known.any():
        return output
    part = output.loc[known, ["chronology_ts", "round_order"]].copy()
    part["event"] = event_key(output.loc[known])
    # Latest timestamp of each round, carried forward through later rounds of the same event.
    latest = part.groupby(["event", "round_order"])["chronology_ts"].max().reset_index()
    latest = latest.sort_values(["event", "round_order"])
    latest["floor"] = latest.groupby("event")["chronology_ts"].cummax()
    latest["floor"] = latest.groupby("event")["floor"].shift(1)
    part = part.merge(latest[["event", "round_order", "floor"]], on=["event", "round_order"], how="left").set_index(part.index)
    adjusted = part["chronology_ts"].where(part["floor"].isna() | (part["chronology_ts"] >= part["floor"]), part["floor"])
    output.loc[known, "chronology_ts"] = adjusted
    return output
