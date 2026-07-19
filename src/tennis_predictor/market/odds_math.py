from __future__ import annotations

import math


def american_to_decimal(odds: float) -> float:
    odds = float(odds)
    if odds == 0:
        raise ValueError("American odds cannot be zero.")
    return 1.0 + (odds / 100.0 if odds > 0 else 100.0 / abs(odds))


def manual_odds_to_decimal(raw_value: str, odds_format: str) -> float:
    """Normalize FanDuel/Polymarket display odds to decimal odds."""
    clean = str(raw_value).strip().replace("$", "").replace("¢", "")
    if not clean:
        raise ValueError("Enter the odds shown in FanDuel or Polymarket.")
    try:
        value = float(clean)
    except ValueError as exc:
        raise ValueError("Odds must be a number, such as -120, +135, 2.10, or 0.58.") from exc

    if odds_format == "American":
        if abs(value) < 100:
            raise ValueError("American odds should look like -120 or +135.")
        return american_to_decimal(value)
    if odds_format == "Decimal":
        if value <= 1.0:
            raise ValueError("Decimal odds must be greater than 1.00.")
        return value
    if odds_format == "Polymarket price":
        # Accept either 0.58 or 58 for a 58-cent contract.
        price = value / 100.0 if value > 1.0 else value
        if not 0.0 < price < 1.0:
            raise ValueError("Polymarket price must be between 0 and 1, or between 1 and 99 cents.")
        return 1.0 / price
    raise ValueError(f"Unsupported odds format: {odds_format}")


def implied_probability(odds: float, fmt: str = "decimal") -> float:
    if fmt == "american":
        odds = american_to_decimal(odds)
    if fmt != "decimal":
        raise ValueError("fmt must be 'decimal' or 'american'.")
    odds = float(odds)
    if not math.isfinite(odds) or odds <= 1.0:
        raise ValueError("Decimal odds must be finite and greater than 1.")
    return 1.0 / odds


def devig_two_way(odds1: float, odds2: float, fmt: str = "decimal") -> tuple[float, float]:
    """Proportional two-way de-vig. Returned probabilities sum to one."""
    p1 = implied_probability(odds1, fmt=fmt)
    p2 = implied_probability(odds2, fmt=fmt)
    total = p1 + p2
    return p1 / total, p2 / total
