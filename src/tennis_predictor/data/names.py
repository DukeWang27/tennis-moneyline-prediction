from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import pandas as pd
from rapidfuzz import fuzz, process


def normalize_name(value: str | None) -> str:
    """Normalize names while preserving enough information to avoid broad merges."""
    if not value:
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold().strip()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # External feeds sometimes use "Nadal R". Reorder only a two-token
    # last-name/initial form; do not attempt aggressive transformations.
    parts = text.split()
    if len(parts) == 2 and len(parts[1]) == 1 and len(parts[0]) > 2:
        text = f"{parts[1]} {parts[0]}"
    return text


@dataclass(frozen=True, slots=True)
class PlayerIdentity:
    player_id: str
    name: str
    normalized_name: str
    tour: str | None = None


class NameResolver:
    """Resolve user/provider names to stable IDs using exact aliases first."""

    def __init__(self, identities: list[PlayerIdentity], threshold: float = 91.0):
        self.identities = identities
        self.threshold = threshold
        self.by_id = {p.player_id: p for p in identities}
        self.by_normalized: dict[str, list[PlayerIdentity]] = {}
        for identity in identities:
            self.by_normalized.setdefault(identity.normalized_name, []).append(identity)
        self.choices = list(self.by_normalized)

    @classmethod
    def from_matches(cls, matches: pd.DataFrame, threshold: float = 91.0) -> "NameResolver":
        records: dict[str, PlayerIdentity] = {}
        for side in ("player1", "player2"):
            id_col = f"{side}_id"
            name_col = side
            for pid, name, tour in matches[[id_col, name_col, "tour"]].dropna().itertuples(index=False):
                pid = str(pid)
                records[pid] = PlayerIdentity(
                    player_id=pid,
                    name=str(name),
                    normalized_name=normalize_name(str(name)),
                    tour=str(tour),
                )
        return cls(list(records.values()), threshold=threshold)


    def register(self, identity: PlayerIdentity) -> None:
        """Add one provider identity without rebuilding the resolver."""
        existing = self.by_id.get(identity.player_id)
        if existing is not None:
            return
        self.identities.append(identity)
        self.by_id[identity.player_id] = identity
        bucket = self.by_normalized.setdefault(identity.normalized_name, [])
        bucket.append(identity)
        if identity.normalized_name not in self.choices:
            self.choices.append(identity.normalized_name)

    def resolve(
        self,
        name: str,
        *,
        tour: str | None = None,
        threshold: float | None = None,
    ) -> PlayerIdentity:
        normalized = normalize_name(name)
        exact = self.by_normalized.get(normalized, [])
        if tour:
            exact = [x for x in exact if x.tour == tour]
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            raise ValueError(f"Ambiguous exact player name {name!r}; specify tour or ID.")

        candidates = self.choices
        result = process.extractOne(normalized, candidates, scorer=fuzz.WRatio)
        if result is None:
            raise KeyError(f"No player match found for {name!r}.")
        matched_name, score, _ = result
        cutoff = self.threshold if threshold is None else threshold
        if score < cutoff:
            raise KeyError(
                f"No sufficiently confident player match for {name!r}; "
                f"best score was {score:.1f}."
            )
        identities = self.by_normalized[matched_name]
        if tour:
            identities = [x for x in identities if x.tour == tour]
        if len(identities) != 1:
            names = ", ".join(f"{x.name} ({x.player_id})" for x in identities)
            raise ValueError(f"Ambiguous fuzzy match for {name!r}: {names}")
        return identities[0]


_INITIALS_SUFFIX = re.compile(r"^(?P<surname>.+?)\s+(?P<initials>(?:[A-Za-z]{1,3}\.[\s-]*)+)$")


def _plain_tokens(value: str) -> list[str]:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).casefold()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return text.split()


def surname_initial_key(abbreviated: str | None) -> str | None:
    """Key for a bookmaker-style name such as ``"Auger-Aliassime F."`` or ``"Kwon S.W."``.

    Returns ``"auger aliassime|f"``: the surname tokens and the first initial. Returns ``None`` when the name
    is not in that form, so callers can fall back to fuzzy matching instead of guessing.
    """
    if not abbreviated or "." not in str(abbreviated):
        return None
    match = _INITIALS_SUFFIX.match(str(abbreviated).strip())
    if not match:
        return None
    surname = _plain_tokens(match.group("surname"))
    initials = _plain_tokens(match.group("initials"))
    if not surname or not initials:
        return None
    return f"{' '.join(surname)}|{initials[0][0]}"


def full_name_keys(full_name: str | None) -> set[str]:
    """Every surname/initial key a full name such as ``"Alex de Minaur"`` could be abbreviated to.

    The split between given names and surname is unknown, so all splits are produced
    (``"de minaur|a"``, ``"minaur|a"``), plus the surname-first order used for some Asian names.
    """
    tokens = _plain_tokens(full_name or "")
    if len(tokens) < 2:
        return set()
    keys = {f"{' '.join(tokens[k:])}|{tokens[0][0]}" for k in range(1, len(tokens))}
    keys.add(f"{tokens[0]}|{tokens[-1][0]}")
    return keys
