from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd

from tennis_predictor.data.names import normalize_name

STARTING_BANKROLL = 100.0
VALID_BET_STATUSES = {"PENDING", "WON", "LOST", "VOID"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def _connection(path: str | Path) -> Iterator[sqlite3.Connection]:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _ensure_columns(connection: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing = {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}
    for name, definition in columns.items():
        if name not in existing:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")


def initialize_paper_trading_db(
    path: str | Path = "data/paper_trading.db",
    *,
    starting_balance: float = STARTING_BANKROLL,
) -> Path:
    if starting_balance <= 0:
        raise ValueError("starting_balance must be positive")
    db_path = Path(path)
    with _connection(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS wallet (
                wallet_id INTEGER PRIMARY KEY CHECK (wallet_id = 1),
                balance REAL NOT NULL CHECK (balance >= 0),
                starting_balance REAL NOT NULL CHECK (starting_balance > 0),
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS bets (
                bet_id INTEGER PRIMARY KEY AUTOINCREMENT,
                fixture_id TEXT NOT NULL UNIQUE,
                matchup TEXT NOT NULL,
                bet_on TEXT NOT NULL,
                odds REAL NOT NULL CHECK (odds > 1.0),
                sportsbook TEXT NOT NULL,
                stake REAL NOT NULL CHECK (stake > 0),
                p_model REAL NOT NULL CHECK (p_model >= 0.0 AND p_model <= 1.0),
                status TEXT NOT NULL CHECK (status IN ('PENDING', 'WON', 'LOST', 'VOID')),
                placed_at TEXT NOT NULL,
                settled_at TEXT
            );

            CREATE TABLE IF NOT EXISTS wallet_adjustments (
                adjustment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                old_balance REAL NOT NULL,
                new_balance REAL NOT NULL,
                note TEXT,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_bets_status ON bets(status);
            """
        )
        _ensure_columns(
            connection,
            "bets",
            {
                "provider_market_id": "TEXT",
                "token_id": "TEXT",
                "opponent": "TEXT",
                "tournament": "TEXT",
                "commence_time": "TEXT",
                "edge": "REAL",
                "kelly_fraction": "REAL",
                "result_match_id": "TEXT",
            },
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO wallet(wallet_id, balance, starting_balance, updated_at)
            VALUES (1, ?, ?, ?)
            """,
            (float(starting_balance), float(starting_balance), _utc_now()),
        )
    return db_path


@dataclass(frozen=True, slots=True)
class WalletSummary:
    balance: float
    starting_balance: float
    wins: int
    losses: int
    voids: int
    pending: int
    settled_stake: float
    settled_profit: float

    @property
    def win_rate(self) -> float:
        decisions = self.wins + self.losses
        return self.wins / decisions if decisions else 0.0

    @property
    def roi(self) -> float:
        return self.settled_profit / self.settled_stake if self.settled_stake else 0.0

    @property
    def bankroll_delta(self) -> float:
        return self.balance - self.starting_balance


def get_wallet_summary(path: str | Path = "data/paper_trading.db") -> WalletSummary:
    initialize_paper_trading_db(path)
    with _connection(path) as connection:
        wallet = connection.execute(
            "SELECT balance, starting_balance FROM wallet WHERE wallet_id = 1"
        ).fetchone()
        aggregate = connection.execute(
            """
            SELECT
                SUM(CASE WHEN status = 'WON' THEN 1 ELSE 0 END) AS wins,
                SUM(CASE WHEN status = 'LOST' THEN 1 ELSE 0 END) AS losses,
                SUM(CASE WHEN status = 'VOID' THEN 1 ELSE 0 END) AS voids,
                SUM(CASE WHEN status = 'PENDING' THEN 1 ELSE 0 END) AS pending,
                COALESCE(SUM(CASE WHEN status IN ('WON','LOST') THEN stake ELSE 0 END), 0) AS settled_stake,
                COALESCE(SUM(
                    CASE
                        WHEN status = 'WON' THEN stake * (odds - 1.0)
                        WHEN status = 'LOST' THEN -stake
                        ELSE 0
                    END
                ), 0) AS settled_profit
            FROM bets
            """
        ).fetchone()
    return WalletSummary(
        balance=float(wallet["balance"]),
        starting_balance=float(wallet["starting_balance"]),
        wins=int(aggregate["wins"] or 0),
        losses=int(aggregate["losses"] or 0),
        voids=int(aggregate["voids"] or 0),
        pending=int(aggregate["pending"] or 0),
        settled_stake=float(aggregate["settled_stake"] or 0.0),
        settled_profit=float(aggregate["settled_profit"] or 0.0),
    )


def set_wallet_balance(
    path: str | Path,
    balance: float,
    *,
    note: str = "Manual bankroll update",
    reset_starting_balance: bool = False,
) -> None:
    initialize_paper_trading_db(path)
    value = round(float(balance), 2)
    if value < 0:
        raise ValueError("Wallet balance cannot be negative")
    with _connection(path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute("SELECT balance FROM wallet WHERE wallet_id = 1").fetchone()
        old = float(row["balance"])
        if reset_starting_balance:
            connection.execute(
                "UPDATE wallet SET balance=?, starting_balance=?, updated_at=? WHERE wallet_id=1",
                (value, value, _utc_now()),
            )
        else:
            connection.execute(
                "UPDATE wallet SET balance=?, updated_at=? WHERE wallet_id=1",
                (value, _utc_now()),
            )
        connection.execute(
            "INSERT INTO wallet_adjustments(old_balance,new_balance,note,created_at) VALUES(?,?,?,?)",
            (old, value, note, _utc_now()),
        )


def list_bets(
    path: str | Path = "data/paper_trading.db",
    *,
    statuses: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    initialize_paper_trading_db(path)
    query = "SELECT * FROM bets"
    params: tuple[str, ...] = ()
    if statuses:
        invalid = set(statuses) - VALID_BET_STATUSES
        if invalid:
            raise ValueError(f"Unsupported bet statuses: {sorted(invalid)}")
        placeholders = ",".join("?" for _ in statuses)
        query += f" WHERE status IN ({placeholders})"
        params = tuple(statuses)
    query += " ORDER BY placed_at DESC, bet_id DESC"
    with _connection(path) as connection:
        return pd.read_sql_query(query, connection, params=params)


def place_paper_bet(
    path: str | Path,
    *,
    fixture_id: str,
    matchup: str,
    bet_on: str,
    odds: float,
    sportsbook: str,
    stake: float,
    p_model: float,
    provider_market_id: str | None = None,
    token_id: str | None = None,
    opponent: str | None = None,
    tournament: str | None = None,
    commence_time: object | None = None,
    edge: float | None = None,
    kelly_fraction: float | None = None,
    result_match_id: str | None = None,
) -> int:
    initialize_paper_trading_db(path)
    odds = float(odds)
    stake = round(float(stake), 2)
    p_model = float(p_model)
    if not fixture_id.strip():
        raise ValueError("fixture_id is required")
    if odds <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.0")
    if stake <= 0:
        raise ValueError("Stake must be positive")
    if not 0.0 <= p_model <= 1.0:
        raise ValueError("p_model must be between 0 and 1")

    commence_iso = None
    if commence_time is not None:
        parsed = pd.to_datetime(commence_time, errors="coerce")
        if pd.notna(parsed):
            commence_iso = pd.Timestamp(parsed).isoformat()

    with _connection(path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        duplicate = connection.execute(
            "SELECT bet_id FROM bets WHERE fixture_id = ?", (fixture_id,)
        ).fetchone()
        if duplicate:
            raise ValueError("A paper bet already exists for this market outcome")
        wallet = connection.execute(
            "SELECT balance FROM wallet WHERE wallet_id = 1"
        ).fetchone()
        balance = float(wallet["balance"])
        if stake > balance + 1e-9:
            raise ValueError(f"Insufficient paper balance: ${balance:.2f} available")
        now = _utc_now()
        cursor = connection.execute(
            """
            INSERT INTO bets(
                fixture_id, matchup, bet_on, odds, sportsbook, stake,
                p_model, status, placed_at, settled_at, provider_market_id,
                token_id, opponent, tournament, commence_time, edge,
                kelly_fraction, result_match_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'PENDING', ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fixture_id,
                matchup,
                bet_on,
                odds,
                sportsbook,
                stake,
                p_model,
                now,
                provider_market_id,
                token_id,
                opponent,
                tournament,
                commence_iso,
                edge,
                kelly_fraction,
                result_match_id,
            ),
        )
        connection.execute(
            "UPDATE wallet SET balance = balance - ?, updated_at = ? WHERE wallet_id = 1",
            (stake, now),
        )
        return int(cursor.lastrowid)


def settle_paper_bet(path: str | Path, *, fixture_id: str, status: str) -> bool:
    status = status.upper()
    if status not in {"WON", "LOST", "VOID"}:
        raise ValueError("Settlement status must be WON, LOST, or VOID")
    initialize_paper_trading_db(path)
    with _connection(path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        bet = connection.execute("SELECT * FROM bets WHERE fixture_id = ?", (fixture_id,)).fetchone()
        if bet is None or bet["status"] != "PENDING":
            return False
        credit = float(bet["stake"]) * float(bet["odds"]) if status == "WON" else float(bet["stake"]) if status == "VOID" else 0.0
        now = _utc_now()
        connection.execute(
            "UPDATE bets SET status = ?, settled_at = ? WHERE fixture_id = ? AND status = 'PENDING'",
            (status, now, fixture_id),
        )
        connection.execute(
            "UPDATE wallet SET balance = balance + ?, updated_at = ? WHERE wallet_id = 1",
            (credit, now),
        )
        return True


def _result_lookup(frame: pd.DataFrame) -> tuple[dict[str, pd.Series], list[pd.Series]]:
    by_id: dict[str, pd.Series] = {}
    rows: list[pd.Series] = []
    for _, row in frame.iterrows():
        fixture_id = str(row.get("fixture_id") or row.get("match_id") or "")
        if fixture_id:
            by_id[fixture_id] = row
        rows.append(row)
    return by_id, rows


def _fallback_result_for_bet(bet: object, rows: list[pd.Series]) -> pd.Series | None:
    bet_on = normalize_name(str(getattr(bet, "bet_on", "")))
    opponent = normalize_name(str(getattr(bet, "opponent", "")))
    commence = pd.to_datetime(getattr(bet, "commence_time", None), errors="coerce")
    candidates: list[tuple[int, pd.Series]] = []
    for row in rows:
        names = {normalize_name(str(row.get("player1", ""))), normalize_name(str(row.get("player2", "")))}
        if bet_on not in names:
            continue
        if opponent and opponent not in names:
            continue
        score = 0
        result_date = pd.to_datetime(row.get("date"), errors="coerce")
        if pd.notna(commence) and pd.notna(result_date):
            day_gap = abs((pd.Timestamp(result_date).normalize() - pd.Timestamp(commence).normalize()).days)
            if day_gap > 2:
                continue
            score += 3 - day_gap
        if normalize_name(str(getattr(bet, "tournament", ""))) == normalize_name(str(row.get("tournament", ""))):
            score += 2
        candidates.append((score, row))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        return None
    return candidates[0][1]


def settle_pending_bets_from_results(path: str | Path, completed_matches: pd.DataFrame) -> dict[str, int]:
    initialize_paper_trading_db(path)
    if completed_matches is None or completed_matches.empty:
        return {"won": 0, "lost": 0, "void": 0}
    frame = completed_matches.copy()
    if "fixture_id" not in frame and "match_id" in frame:
        frame["fixture_id"] = frame["match_id"].astype(str)
    required = {"player1", "player2", "player1_won"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Completed results are missing columns: {sorted(missing)}")
    pending = list_bets(path, statuses=("PENDING",))
    if pending.empty:
        return {"won": 0, "lost": 0, "void": 0}
    by_id, rows = _result_lookup(frame)
    counts = {"won": 0, "lost": 0, "void": 0}
    for bet in pending.itertuples(index=False):
        result = by_id.get(str(getattr(bet, "result_match_id", "") or ""))
        if result is None:
            result = by_id.get(str(bet.fixture_id))
        if result is None:
            result = _fallback_result_for_bet(bet, rows)
        if result is None:
            continue
        if bool(result.get("void", False)) or bool(result.get("walkover", False)):
            status = "VOID"
        else:
            winner = str(result["player1"] if int(result["player1_won"]) == 1 else result["player2"])
            status = "WON" if normalize_name(str(bet.bet_on)) == normalize_name(winner) else "LOST"
        if settle_paper_bet(path, fixture_id=str(bet.fixture_id), status=status):
            counts[status.casefold()] += 1
    return counts
