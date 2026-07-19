from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests

from tennis_predictor.data.names import normalize_name
from tennis_predictor.market.odds_math import devig_two_way


QUOTE_COLUMNS = [
    "provider_event_id",
    "snapshot_time",
    "commence_time",
    "tour",
    "tournament",
    "player1",
    "player2",
    "bookmaker",
    "bookmaker_key",
    "decimal_odds1",
    "decimal_odds2",
    "polymarket_price1",
    "polymarket_price2",
    "best_bid1",
    "best_bid2",
    "spread1",
    "spread2",
    "liquidity",
    "source",
    "market_id",
    "market_slug",
    "market_url",
    "token_id1",
    "token_id2",
    "is_closing",
]


LIVE_MARKET_COLUMNS = [
    "market_id", "condition_id", "provider_event_id", "question",
    "display_title", "commence_time", "tournament", "tour", "tier",
    "player1", "player2", "token_id1", "token_id2",
    "gamma_price1", "gamma_price2", "best_bid1", "best_bid2",
    "best_ask1", "best_ask2", "spread1", "spread2",
    "liquidity1", "liquidity2", "liquidity", "market_slug",
    "market_url", "parse_status", "parse_reason", "active",
    "fetched_at", "raw_json",
]


def _empty_live_markets() -> pd.DataFrame:
    return pd.DataFrame(columns=LIVE_MARKET_COLUMNS)


def _infer_tour_tier(text: str) -> tuple[str, str]:
    normalized = normalize_name(text)
    if "wta" in normalized or "women" in normalized:
        tour = "WTA"
    elif "atp" in normalized or "men" in normalized or "challenger" in normalized:
        tour = "ATP"
    else:
        tour = ""
    if "itf" in normalized or re.search(r"\b[wm](?:15|25|35|50|75|100)\b", normalized):
        tier = "ITF"
    elif "challenger" in normalized:
        tier = "Challenger"
    elif any(name in normalized for name in (
        "wimbledon", "us open", "australian open", "roland garros", "french open"
    )):
        tier = "Slam"
    elif "masters" in normalized or "1000" in normalized:
        tier = "Masters"
    elif "500" in normalized:
        tier = "500"
    elif "250" in normalized:
        tier = "250"
    else:
        tier = "Other"
    return tour, tier


def _empty_quotes() -> pd.DataFrame:
    return pd.DataFrame(columns=QUOTE_COLUMNS)


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def _clean_player_name(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip(" -:?\t\n")
    text = re.sub(
        r"\s+(?:moneyline|match winner|winner|to win)(?:\?|$).*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text).strip()
    text = re.sub(
        r"\s*[-|]\s*(?:ATP|WTA|ITF|Challenger|Tennis)\b.*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    return text


def _matchup_from_text(text: str) -> tuple[str, str] | None:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    parts = re.split(r"\s+(?:vs\.?|v\.|versus)\s+", clean, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) != 2:
        return None
    left, right = parts
    # Polymarket sports titles sometimes prefix the tournament before a colon.
    if ":" in left:
        prefix, candidate = left.rsplit(":", 1)
        if re.search(r"\b(?:atp|wta|itf|challenger|tennis)\b", prefix, re.IGNORECASE):
            left = candidate
    left = _clean_player_name(left)
    right = _clean_player_name(right)
    if not left or not right:
        return None
    return left, right


def _yes_player(question: str, market: dict[str, Any]) -> str | None:
    patterns = [
        r"will\s+(.+?)\s+(?:win|beat|defeat)",
        r"does\s+(.+?)\s+(?:win|beat|defeat)",
        r"^(.+?)\s+to\s+win",
    ]
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            return _clean_player_name(match.group(1))
    group_item = str(market.get("groupItemTitle") or market.get("shortTitle") or "")
    cleaned = _clean_player_name(group_item)
    return cleaned or None


def _event_text(event: dict[str, Any], market: dict[str, Any] | None = None) -> str:
    pieces: list[str] = []
    for item in (event, market or {}):
        for key in (
            "title",
            "subtitle",
            "question",
            "description",
            "slug",
            "seriesTitle",
            "category",
            "subcategory",
            "sportsMarketType",
            "groupItemTitle",
        ):
            value = item.get(key)
            if value:
                pieces.append(str(value))
        for relation in ("tags", "series", "events"):
            value = item.get(relation)
            if isinstance(value, list):
                for nested in value:
                    if isinstance(nested, dict):
                        pieces.extend(
                            str(nested.get(key) or "")
                            for key in ("label", "name", "title", "slug")
                        )
                    else:
                        pieces.append(str(nested))
    return " ".join(pieces)


def _looks_like_tennis(event: dict[str, Any], market: dict[str, Any] | None = None) -> bool:
    text = _event_text(event, market).casefold()
    return bool(re.search(r"\b(?:tennis|atp|wta|itf|challenger)\b", text))


def _market_active(event: dict[str, Any], market: dict[str, Any]) -> bool:
    if _as_bool(event.get("closed")) or _as_bool(market.get("closed")):
        return False
    if event.get("active") is not None and not _as_bool(event.get("active")):
        return False
    if market.get("active") is not None and not _as_bool(market.get("active")):
        return False
    return True


def _event_for_market(market: dict[str, Any]) -> dict[str, Any]:
    events = market.get("events")
    if isinstance(events, list) and events and isinstance(events[0], dict):
        return events[0]
    return {}


def _event_identity(event: dict[str, Any], market: dict[str, Any] | None = None) -> str:
    market = market or {}
    return str(
        event.get("id")
        or event.get("slug")
        or market.get("eventId")
        or market.get("id")
        or market.get("slug")
        or ""
    )


def _market_time(event: dict[str, Any], market: dict[str, Any]) -> pd.Timestamp:
    value = (
        market.get("gameStartTime")
        or market.get("eventStartTime")
        or event.get("gameStartTime")
        or event.get("startDate")
        or market.get("startDate")
        or market.get("endDate")
        or event.get("endDate")
    )
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if timestamp is None or pd.isna(timestamp):
        return pd.NaT
    return timestamp.tz_convert(None)


def _series_title(event: dict[str, Any], market: dict[str, Any]) -> str:
    direct = event.get("seriesTitle") or market.get("seriesTitle")
    if direct:
        return str(direct)
    series = event.get("series")
    if isinstance(series, list) and series and isinstance(series[0], dict):
        title = series[0].get("title") or series[0].get("name")
        if title:
            return str(title)
    title = str(event.get("title") or market.get("question") or "Polymarket Tennis")
    if ":" in title:
        prefix = title.split(":", 1)[0].strip()
        if prefix:
            return prefix
    return title


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


@dataclass(slots=True)
class PolymarketClient:
    """Discover active tennis winner markets and read executable CLOB prices."""

    gamma_url: str = "https://gamma-api.polymarket.com"
    clob_url: str = "https://clob.polymarket.com"
    timeout_seconds: int = 5
    search_queries: tuple[str, ...] = (
        "tennis",
        "ITF tennis",
        "ATP Challenger",
    )
    last_diagnostics: dict[str, Any] = field(default_factory=dict, init=False)

    def _request_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> Any:
        response = requests.get(url, params=params, timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.json()

    def _search_events(self, *, limit: int, pages: int) -> list[dict[str, Any]]:
        """Target tennis through Gamma public search without failing the full refresh.

        Public search is intentionally used as a targeted discovery source. One
        failed query or page is recorded in diagnostics and the remaining
        queries continue, so a temporary Gamma validation error cannot erase
        markets already found.
        """
        found: dict[str, dict[str, Any]] = {}
        query_counts: dict[str, int] = {}
        errors: list[str] = []
        per_page = min(max(int(limit), 1), 100)
        for query in self.search_queries:
            count = 0
            for page in range(1, max(1, int(pages)) + 1):
                try:
                    payload = self._request_json(
                        f"{self.gamma_url}/public-search",
                        params={
                            "q": query,
                            "events_status": "active",
                            "limit_per_type": per_page,
                            "page": page,
                            "keep_closed_markets": 0,
                            "search_tags": "true",
                            "search_profiles": "false",
                        },
                    )
                except Exception as exc:
                    errors.append(f"public-search {query!r} page {page}: {exc}")
                    break
                events = payload.get("events", []) if isinstance(payload, dict) else []
                if not isinstance(events, list) or not events:
                    break
                for event in events:
                    if not isinstance(event, dict):
                        continue
                    identity = _event_identity(event)
                    if identity:
                        tagged_event = dict(event)
                        tagged_event["_tennis_search_hit"] = True
                        found[identity] = tagged_event
                        count += 1
                pagination = payload.get("pagination", {}) if isinstance(payload, dict) else {}
                total_pages = (
                    _safe_float(pagination.get("totalPages"))
                    if isinstance(pagination, dict)
                    else None
                )
                if total_pages is not None and page >= int(total_pages):
                    break
                if len(events) < per_page:
                    break
            query_counts[query] = count
        self.last_diagnostics["search_query_counts"] = query_counts
        if errors:
            self.last_diagnostics.setdefault("discovery_errors", []).extend(errors)
        return list(found.values())

    def _keyset_markets(
        self,
        *,
        limit: int,
        max_pages: int,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
    ) -> list[dict[str, Any]]:
        """Read stable Gamma pages with cursors instead of fragile large offsets."""
        rows: list[dict[str, Any]] = []
        cursor: str | None = None
        pages_read = 0
        errors: list[str] = []
        seen_cursors: set[str] = set()
        for page in range(max(1, int(max_pages))):
            params: dict[str, Any] = {
                "limit": min(max(int(limit), 1), 100),
                "closed": "false",
                "include_tag": "true",
                "ascending": "true",
                "order": "start_date",
            }
            if cursor:
                params["after_cursor"] = cursor
            if start_date is not None:
                start = pd.to_datetime(start_date, utc=True, errors="coerce")
                if pd.notna(start):
                    params["start_date_min"] = pd.Timestamp(start).isoformat()
            if end_date is not None:
                end = pd.to_datetime(end_date, utc=True, errors="coerce")
                if pd.notna(end):
                    params["start_date_max"] = pd.Timestamp(end).isoformat()
            try:
                payload = self._request_json(
                    f"{self.gamma_url}/markets/keyset",
                    params=params,
                )
            except Exception as exc:
                errors.append(f"markets/keyset page {page + 1}: {exc}")
                break
            markets = payload.get("markets", []) if isinstance(payload, dict) else []
            if not isinstance(markets, list) or not markets:
                break
            pages_read += 1
            for market in markets:
                if not isinstance(market, dict):
                    continue
                event = _event_for_market(market)
                if _market_active(event, market) and _looks_like_tennis(event, market):
                    rows.append(market)
            next_cursor = str(payload.get("next_cursor") or "") if isinstance(payload, dict) else ""
            if not next_cursor or next_cursor in seen_cursors:
                break
            seen_cursors.add(next_cursor)
            cursor = next_cursor
        self.last_diagnostics["keyset_pages_read"] = pages_read
        self.last_diagnostics["keyset_tennis_markets"] = len(rows)
        if errors:
            self.last_diagnostics.setdefault("discovery_errors", []).extend(errors)
        return rows

    def _legacy_market_page(self, *, limit: int) -> list[dict[str, Any]]:
        """Compatibility fallback: read only the first offset page.

        This never advances to the large offsets that caused the observed 422.
        """
        try:
            payload = self._request_json(
                f"{self.gamma_url}/markets",
                params={
                    "active": "true",
                    "closed": "false",
                    "limit": min(max(int(limit), 1), 100),
                    "offset": 0,
                    "include_tag": "true",
                },
            )
        except Exception as exc:
            self.last_diagnostics.setdefault("discovery_errors", []).append(
                f"markets first page: {exc}"
            )
            return []
        if not isinstance(payload, list):
            return []
        rows = []
        for market in payload:
            if not isinstance(market, dict):
                continue
            event = _event_for_market(market)
            if _market_active(event, market) and _looks_like_tennis(event, market):
                rows.append(market)
        self.last_diagnostics["legacy_first_page_tennis_markets"] = len(rows)
        return rows

    @staticmethod
    def _iter_event_markets(events: Iterable[dict[str, Any]]) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
        for event in events:
            markets = event.get("markets", [])
            if not isinstance(markets, list):
                continue
            for market in markets:
                if isinstance(market, dict):
                    yield event, market

    def _discover(
        self,
        *,
        limit: int,
        search_pages: int,
        keyset_pages: int,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        union_keyset: bool = False,
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """Combine targeted search with cursor discovery and a safe fallback."""
        pairs: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        if union_keyset:
            for market in self._legacy_market_page(limit=limit):
                event = _event_for_market(market)
                key = str(
                    market.get("id")
                    or market.get("slug")
                    or _event_identity(event, market)
                )
                if key:
                    pairs[key] = (event, market)
        events = self._search_events(limit=limit, pages=search_pages)
        for event, market in self._iter_event_markets(events):
            if not _market_active(event, market):
                continue
            key = str(market.get("id") or market.get("slug") or _event_identity(event, market))
            if key:
                pairs[key] = (event, market)
        # Cursor scanning is a fallback, not a mandatory full-catalog crawl.
        # Targeted public search and the first active page are usually enough;
        # keyset is used only when both return nothing.
        if not pairs:
            for market in self._keyset_markets(
                limit=limit,
                max_pages=keyset_pages,
                start_date=start_date,
                end_date=end_date,
            ):
                event = _event_for_market(market)
                key = str(market.get("id") or market.get("slug") or _event_identity(event, market))
                if key:
                    pairs[key] = (event, market)
        if not pairs:
            for market in self._legacy_market_page(limit=limit):
                event = _event_for_market(market)
                key = str(market.get("id") or market.get("slug") or _event_identity(event, market))
                if key:
                    pairs[key] = (event, market)
        self.last_diagnostics["discovered_market_pairs"] = len(pairs)
        return list(pairs.values())

    def order_book(self, token_id: str) -> dict[str, Any]:
        try:
            response = requests.get(
                f"{self.clob_url}/book",
                params={"token_id": token_id},
                timeout=self.timeout_seconds,
            )
            if response.status_code >= 400:
                return {}
            payload = response.json()
        except (requests.RequestException, ValueError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def order_books(
        self,
        token_ids: Iterable[str],
        *,
        batch_size: int = 50,
    ) -> dict[str, dict[str, Any]]:
        """Fetch multiple CLOB books in bounded POST batches.

        Polymarket exposes ``POST /books`` for batch order-book reads. A failed
        batch falls back to individual GET requests so the scanner still works
        with older mocks or temporary endpoint issues.
        """
        unique = list(dict.fromkeys(str(token).strip() for token in token_ids if str(token).strip()))
        if not unique:
            return {}
        output: dict[str, dict[str, Any]] = {}
        size = max(1, min(int(batch_size), 100))
        for start in range(0, len(unique), size):
            batch = unique[start : start + size]
            try:
                response = requests.post(
                    f"{self.clob_url}/books",
                    json=[{"token_id": token} for token in batch],
                    timeout=self.timeout_seconds,
                    headers={"Content-Type": "application/json"},
                )
                if response.status_code >= 400:
                    raise requests.RequestException(f"HTTP {response.status_code}")
                payload = response.json()
                if not isinstance(payload, list):
                    raise ValueError("Unexpected /books response")
                for book in payload:
                    if not isinstance(book, dict):
                        continue
                    token = str(book.get("asset_id") or book.get("token_id") or "").strip()
                    if token:
                        output[token] = book
            except (requests.RequestException, ValueError, AttributeError):
                for token in batch:
                    output[token] = self.order_book(token)
        return output

    @staticmethod
    def _book_values(book: dict[str, Any]) -> tuple[float | None, float | None, float | None, float]:
        bids = [
            (_safe_float(row.get("price")), _safe_float(row.get("size")))
            for row in book.get("bids", [])
            if isinstance(row, dict)
        ]
        asks = [
            (_safe_float(row.get("price")), _safe_float(row.get("size")))
            for row in book.get("asks", [])
            if isinstance(row, dict)
        ]
        valid_bids = [(price, size) for price, size in bids if price is not None]
        valid_asks = [(price, size) for price, size in asks if price is not None]
        best_bid = max((price for price, _ in valid_bids), default=None)
        best_ask = min((price for price, _ in valid_asks), default=None)
        spread = (
            best_ask - best_bid
            if best_ask is not None and best_bid is not None
            else None
        )
        ask_liquidity = sum((size or 0.0) * price for price, size in valid_asks)
        return best_bid, best_ask, spread, ask_liquidity

    def _parse_market(
        self,
        event: dict[str, Any],
        market: dict[str, Any],
        *,
        use_order_books: bool,
    ) -> dict[str, Any] | None:
        if not _market_active(event, market):
            return None
        if not event.get("_tennis_search_hit") and not _looks_like_tennis(event, market):
            return None
        outcomes = [str(value) for value in _json_list(market.get("outcomes"))]
        prices_raw = _json_list(market.get("outcomePrices"))
        tokens = [str(value) for value in _json_list(market.get("clobTokenIds"))]
        if len(outcomes) != 2 or len(prices_raw) != 2:
            return None
        prices = [_safe_float(value) for value in prices_raw]
        if any(value is None for value in prices):
            return None
        event_title = str(event.get("title") or event.get("question") or "")
        question = str(market.get("question") or event_title)
        players: tuple[str, str] | None = None
        side_prices: list[float] | None = None
        side_tokens: list[str | None] = [None, None]
        if {item.casefold() for item in outcomes} == {"yes", "no"}:
            matchup = _matchup_from_text(event_title) or _matchup_from_text(question)
            yes_name = _yes_player(question, market)
            if matchup and yes_name:
                first, second = matchup
                lookup = {outcome.casefold(): index for index, outcome in enumerate(outcomes)}
                yes_index, no_index = lookup["yes"], lookup["no"]
                if normalize_name(yes_name) == normalize_name(first):
                    players = (first, second)
                    side_prices = [float(prices[yes_index]), float(prices[no_index])]
                    if len(tokens) == 2:
                        side_tokens = [tokens[yes_index], tokens[no_index]]
                elif normalize_name(yes_name) == normalize_name(second):
                    players = (first, second)
                    side_prices = [float(prices[no_index]), float(prices[yes_index])]
                    if len(tokens) == 2:
                        side_tokens = [tokens[no_index], tokens[yes_index]]
        else:
            players = (_clean_player_name(outcomes[0]), _clean_player_name(outcomes[1]))
            side_prices = [float(prices[0]), float(prices[1])]
            if len(tokens) == 2:
                side_tokens = [tokens[0], tokens[1]]
        if players is None or side_prices is None or not all(players):
            return None

        executable = list(side_prices)
        best_bids: list[float | None] = [None, None]
        spreads: list[float | None] = [None, None]
        liquidity = _safe_float(market.get("liquidityNum") or market.get("liquidity")) or 0.0
        if use_order_books:
            book_liquidity = 0.0
            for index, token_id in enumerate(side_tokens):
                if not token_id:
                    continue
                bid, ask, spread, side_liquidity = self._book_values(self.order_book(token_id))
                best_bids[index] = bid
                spreads[index] = spread
                book_liquidity += side_liquidity
                if ask is not None:
                    executable[index] = ask
            if book_liquidity > 0:
                liquidity = book_liquidity
        if any(price <= 0 or price >= 1 for price in executable):
            return None

        slug = str(event.get("slug") or market.get("slug") or "")
        event_id = _event_identity(event, market)
        return {
            "provider_event_id": event_id,
            "snapshot_time": pd.Timestamp.now(tz="UTC").tz_convert(None),
            "commence_time": _market_time(event, market),
            "tour": "",
            "tournament": _series_title(event, market),
            "player1": players[0],
            "player2": players[1],
            "bookmaker": "Polymarket",
            "bookmaker_key": "polymarket",
            "decimal_odds1": 1.0 / executable[0],
            "decimal_odds2": 1.0 / executable[1],
            "polymarket_price1": executable[0],
            "polymarket_price2": executable[1],
            "best_bid1": best_bids[0],
            "best_bid2": best_bids[1],
            "spread1": spreads[0],
            "spread2": spreads[1],
            "liquidity": liquidity,
            "source": "polymarket",
            "market_id": str(market.get("id") or market.get("conditionId") or ""),
            "market_slug": slug,
            "market_url": f"https://polymarket.com/event/{slug}" if slug else "",
            "token_id1": side_tokens[0],
            "token_id2": side_tokens[1],
            "is_closing": False,
        }

    def fetch_tennis_quotes(
        self,
        *,
        use_order_books: bool = False,
        limit: int = 100,
        search_pages: int = 2,
        keyset_pages: int = 3,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        self.last_diagnostics.clear()
        pairs = self._discover(
            limit=limit,
            search_pages=search_pages,
            keyset_pages=keyset_pages,
        )
        rows = [
            row
            for event, market in pairs
            if (row := self._parse_market(event, market, use_order_books=use_order_books))
            is not None
        ]
        self.last_diagnostics["parsed_quote_rows"] = len(rows)
        if not rows:
            return _empty_quotes()
        frame = pd.DataFrame(rows, columns=QUOTE_COLUMNS)
        if start_date is not None or end_date is not None:
            times = pd.to_datetime(frame["commence_time"], errors="coerce")
            mask = pd.Series(True, index=frame.index)
            if start_date is not None:
                lower = pd.Timestamp(start_date).normalize() - pd.Timedelta(days=1)
                mask &= times.isna() | (times >= lower)
            if end_date is not None:
                upper = pd.Timestamp(end_date).normalize() + pd.Timedelta(days=2)
                mask &= times.isna() | (times < upper)
            frame = frame[mask].copy()
        return frame.drop_duplicates(
            subset=["provider_event_id", "market_id", "player1", "player2"],
            keep="last",
        ).reset_index(drop=True)


    def fetch_active_tennis_markets(
        self,
        *,
        include_books: bool = True,
        page_size: int = 100,
        max_pages: int = 4,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        max_markets: int = 60,
        book_batch_size: int = 50,
    ) -> pd.DataFrame:
        """Discover a bounded set of active tennis markets for one time window.

        The previous scanner could crawl dozens of Gamma pages and fetch books
        for markets outside the selected day. This implementation uses one page
        per targeted search, falls back to at most a few keyset pages, filters
        by time *before* CLOB calls, caps the rendered board, and batches books.
        """
        self.last_diagnostics.clear()
        pairs = self._discover(
            limit=page_size,
            search_pages=1,
            keyset_pages=min(max(int(max_pages), 1), 4),
            start_date=start_date,
            end_date=end_date,
            union_keyset=True,
        )
        raw_by_id: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        for event, market in pairs:
            if not _market_active(event, market):
                continue
            identity = str(
                market.get("id")
                or market.get("conditionId")
                or market.get("slug")
                or _event_identity(event, market)
                or ""
            )
            if identity:
                raw_by_id[identity] = (event, market)

        rows: list[dict[str, Any]] = []
        now = pd.Timestamp.now(tz="UTC").tz_convert(None)
        for market_id, (event, market) in raw_by_id.items():
            parsed = self._parse_market(event, market, use_order_books=False)
            question = str(
                market.get("question")
                or event.get("title")
                or "Polymarket tennis market"
            )
            display_title = str(event.get("title") or question)
            tournament = _series_title(event, market)
            text = _event_text(event, market)
            tour, tier = _infer_tour_tier(text)
            outcomes = [str(value) for value in _json_list(market.get("outcomes"))]
            tokens = [str(value) for value in _json_list(market.get("clobTokenIds"))]
            prices = [_safe_float(value) for value in _json_list(market.get("outcomePrices"))]
            slug = str(event.get("slug") or market.get("slug") or "")

            if parsed is not None:
                player1 = str(parsed["player1"])
                player2 = str(parsed["player2"])
                token1 = str(parsed.get("token_id1") or "")
                token2 = str(parsed.get("token_id2") or "")
                gamma1 = float(parsed["polymarket_price1"])
                gamma2 = float(parsed["polymarket_price2"])
                parse_status = "PARSED"
                parse_reason = "Two player outcomes resolved."
            else:
                matchup = _matchup_from_text(display_title) or _matchup_from_text(question)
                if matchup:
                    player1, player2 = matchup
                elif len(outcomes) == 2 and {value.casefold() for value in outcomes} != {"yes", "no"}:
                    player1, player2 = _clean_player_name(outcomes[0]), _clean_player_name(outcomes[1])
                else:
                    player1, player2 = "", ""
                token1 = tokens[0] if len(tokens) > 0 else ""
                token2 = tokens[1] if len(tokens) > 1 else ""
                gamma1 = prices[0] if len(prices) > 0 and prices[0] is not None else np.nan
                gamma2 = prices[1] if len(prices) > 1 and prices[1] is not None else np.nan
                parse_status = "NEEDS MAPPING"
                parse_reason = "The market is tennis, but its two player outcomes could not be parsed safely."

            compact_raw = {
                "id": market_id,
                "event_id": _event_identity(event, market),
                "question": question,
                "outcomes": outcomes,
                "clobTokenIds": tokens,
                "outcomePrices": prices,
            }
            rows.append(
                {
                    "market_id": market_id,
                    "condition_id": str(market.get("conditionId") or ""),
                    "provider_event_id": _event_identity(event, market),
                    "question": question,
                    "display_title": display_title,
                    "commence_time": _market_time(event, market),
                    "tournament": tournament,
                    "tour": tour,
                    "tier": tier,
                    "player1": player1,
                    "player2": player2,
                    "token_id1": token1,
                    "token_id2": token2,
                    "gamma_price1": gamma1,
                    "gamma_price2": gamma2,
                    "best_bid1": np.nan,
                    "best_bid2": np.nan,
                    "best_ask1": np.nan,
                    "best_ask2": np.nan,
                    "spread1": np.nan,
                    "spread2": np.nan,
                    "liquidity1": 0.0,
                    "liquidity2": 0.0,
                    "liquidity": _safe_float(market.get("liquidityNum") or market.get("liquidity")) or 0.0,
                    "market_slug": slug,
                    "market_url": f"https://polymarket.com/event/{slug}" if slug else "",
                    "parse_status": parse_status,
                    "parse_reason": parse_reason,
                    "active": True,
                    "fetched_at": now,
                    "raw_json": compact_raw,
                }
            )

        if not rows:
            self.last_diagnostics.update(
                {"active_tennis_markets_raw": 0, "parsed_tennis_markets": 0, "books_requested": 0}
            )
            return _empty_live_markets()

        frame = pd.DataFrame(rows)
        self.last_diagnostics["markets_before_date_filter"] = len(frame)
        times = pd.to_datetime(frame["commence_time"], errors="coerce", utc=True)
        mask = times.notna()
        if start_date is not None:
            lower = pd.to_datetime(start_date, errors="coerce", utc=True)
            if pd.notna(lower):
                mask &= times >= lower
        if end_date is not None:
            upper = pd.to_datetime(end_date, errors="coerce", utc=True)
            if pd.notna(upper):
                mask &= times < upper
        frame = frame[mask].copy()
        frame["commence_time"] = times[mask].dt.tz_convert(None)
        frame = frame.sort_values(["commence_time", "tournament", "display_title"])
        available = len(frame)
        cap = max(1, int(max_markets))
        if available > cap:
            frame = frame.head(cap).copy()
        self.last_diagnostics["markets_after_date_filter"] = available
        self.last_diagnostics["markets_truncated"] = max(available - len(frame), 0)

        if include_books and not frame.empty:
            locations: dict[str, list[tuple[int, int]]] = {}
            for index, row in frame.iterrows():
                for side in (1, 2):
                    token = str(row.get(f"token_id{side}") or "").strip()
                    if token:
                        locations.setdefault(token, []).append((index, side))
            books = self.order_books(locations.keys(), batch_size=book_batch_size)
            for token, book in books.items():
                bid, ask, spread, visible = self._book_values(book)
                for index, side in locations.get(token, []):
                    frame.at[index, f"best_bid{side}"] = bid
                    frame.at[index, f"best_ask{side}"] = ask
                    frame.at[index, f"spread{side}"] = spread
                    frame.at[index, f"liquidity{side}"] = visible
            visible_total = frame[["liquidity1", "liquidity2"]].sum(axis=1)
            frame["liquidity"] = visible_total.where(visible_total > 0, frame["liquidity"])

        self.last_diagnostics.update(
            {
                "active_tennis_markets_raw": len(raw_by_id),
                "parsed_tennis_markets": int(frame["parse_status"].eq("PARSED").sum()) if not frame.empty else 0,
                "books_requested": int(
                    frame[["token_id1", "token_id2"]]
                    .astype(str)
                    .apply(lambda col: col.str.len().gt(0))
                    .sum()
                    .sum()
                    if include_books and not frame.empty
                    else 0
                ),
            }
        )
        for column in LIVE_MARKET_COLUMNS:
            if column not in frame:
                frame[column] = np.nan
        return frame[LIVE_MARKET_COLUMNS].drop_duplicates("market_id", keep="last").reset_index(drop=True)

    def enrich_order_books(self, quotes: pd.DataFrame) -> pd.DataFrame:
        """Fetch CLOB books only for matched markets, concurrently."""
        if quotes.empty:
            return quotes.copy()
        enriched = quotes.copy()
        locations: dict[str, list[tuple[Any, int]]] = {}
        for index, row in enriched.iterrows():
            for side in (1, 2):
                token_id = str(row.get(f"token_id{side}") or "").strip()
                if token_id:
                    locations.setdefault(token_id, []).append((index, side))
        if not locations:
            return enriched
        workers = min(8, len(locations))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            books = dict(zip(locations, executor.map(self.order_book, locations)))
        liquidity_by_row: dict[Any, float] = {}
        for token_id, book in books.items():
            bid, ask, spread, liquidity = self._book_values(book)
            for index, side in locations[token_id]:
                liquidity_by_row[index] = liquidity_by_row.get(index, 0.0) + liquidity
                enriched.at[index, f"best_bid{side}"] = bid
                enriched.at[index, f"spread{side}"] = spread
                if ask is not None and 0 < ask < 1:
                    enriched.at[index, f"polymarket_price{side}"] = ask
                    enriched.at[index, f"decimal_odds{side}"] = 1.0 / ask
        for index, liquidity in liquidity_by_row.items():
            if liquidity > 0:
                enriched.at[index, "liquidity"] = liquidity
        return enriched

class LiveOddsMatcher:
    """Align upcoming provider quotes to normalized fixtures without mutating fixtures."""

    def __init__(self, threshold: float = 84.0):
        self.threshold = threshold

    @staticmethod
    def _name_score(a: str, b: str) -> float:
        from rapidfuzz.fuzz import WRatio

        return float(WRatio(normalize_name(a), normalize_name(b)))

    def align(self, fixtures: pd.DataFrame, quotes: pd.DataFrame) -> pd.DataFrame:
        columns = [
            "fixture_id", "fixture_player1", "fixture_player2", "alignment_score",
            "provider_event_id", "snapshot_time", "commence_time", "tour",
            "tournament", "player1", "player2", "bookmaker", "bookmaker_key",
            "decimal_odds1", "decimal_odds2", "polymarket_price1",
            "polymarket_price2", "best_bid1", "best_bid2", "spread1",
            "spread2", "liquidity", "source", "market_id", "market_slug",
            "market_url", "token_id1", "token_id2", "is_closing",
        ]
        if fixtures.empty or quotes.empty:
            return pd.DataFrame(columns=columns)
        rows = []
        for quote in quotes.itertuples(index=False):
            commence = pd.Timestamp(quote.commence_time) if pd.notna(quote.commence_time) else pd.NaT
            candidates = fixtures.copy()
            if pd.notna(commence):
                candidate_dates = pd.to_datetime(candidates["commence_time"], errors="coerce")
                candidates = candidates[(candidate_dates - commence).abs() <= pd.Timedelta(days=2)]
            best = None
            for fixture in candidates.itertuples(index=False):
                direct = (self._name_score(quote.player1, fixture.player1) + self._name_score(quote.player2, fixture.player2)) / 2
                reverse = (self._name_score(quote.player1, fixture.player2) + self._name_score(quote.player2, fixture.player1)) / 2
                reversed_order = reverse > direct
                score = max(direct, reverse)
                if best is None or score > best[0]:
                    best = (score, fixture, reversed_order)
            if best is None or best[0] < self.threshold:
                continue
            score, fixture, reversed_order = best
            data = quote._asdict()
            if reversed_order:
                swap_pairs = (
                    ("decimal_odds1", "decimal_odds2"),
                    ("polymarket_price1", "polymarket_price2"),
                    ("best_bid1", "best_bid2"),
                    ("spread1", "spread2"),
                    ("token_id1", "token_id2"),
                )
                for first, second in swap_pairs:
                    if first in data or second in data:
                        data[first], data[second] = data.get(second), data.get(first)
            data.update(
                {
                    "fixture_id": str(fixture.fixture_id),
                    "fixture_player1": str(fixture.player1),
                    "fixture_player2": str(fixture.player2),
                    "alignment_score": score,
                }
            )
            rows.append(data)
        return pd.DataFrame(rows)


class BestPriceEngine:
    def __init__(self, sharp_bookmakers: tuple[str, ...] = ("pinnacle", "pinnaclesports", "circa")):
        self.sharp_bookmakers = tuple(x.casefold() for x in sharp_bookmakers)

    def summarize(self, aligned_quotes: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
        if predictions.empty:
            return predictions.copy()
        output = predictions.copy()
        if aligned_quotes.empty:
            for column in (
                "best_odds1", "best_book1", "best_odds2", "best_book2",
                "fair_market_p1", "fair_market_p2", "edge1", "edge2", "sharp_book",
            ):
                output[column] = np.nan
            return output
        summaries = []
        for fixture_id, group in aligned_quotes.groupby("fixture_id", sort=False):
            clean = group.dropna(subset=["decimal_odds1", "decimal_odds2"]).copy()
            if clean.empty:
                continue
            best1 = clean.loc[clean["decimal_odds1"].astype(float).idxmax()]
            best2 = clean.loc[clean["decimal_odds2"].astype(float).idxmax()]
            clean["_key"] = clean.get("bookmaker_key", clean["bookmaker"]).astype(str).str.casefold()
            sharp = clean[clean["_key"].isin(self.sharp_bookmakers)]
            if sharp.empty:
                implied_total = 1.0 / clean["decimal_odds1"].astype(float) + 1.0 / clean["decimal_odds2"].astype(float)
                market_row = clean.loc[implied_total.idxmin()]
            else:
                market_row = sharp.iloc[0]
            fair1, fair2 = devig_two_way(
                float(market_row["decimal_odds1"]), float(market_row["decimal_odds2"])
            )
            summaries.append(
                {
                    "fixture_id": str(fixture_id),
                    "best_odds1": float(best1["decimal_odds1"]),
                    "best_book1": str(best1["bookmaker"]),
                    "best_odds2": float(best2["decimal_odds2"]),
                    "best_book2": str(best2["bookmaker"]),
                    "fair_market_p1": fair1,
                    "fair_market_p2": fair2,
                    "sharp_book": str(market_row["bookmaker"]),
                }
            )
        summary = pd.DataFrame(summaries)
        output = output.merge(summary, on="fixture_id", how="left")
        output["edge1"] = output["p_model1"] - output["fair_market_p1"]
        output["edge2"] = (1.0 - output["p_model1"]) - output["fair_market_p2"]
        return output
