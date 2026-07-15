from __future__ import annotations

import math


def american_to_decimal(odds: float) -> float:
    odds = float(odds)
    if odds == 0:
        raise ValueError("American odds cannot be zero.")
    return 1.0 + (odds / 100.0 if odds > 0 else 100.0 / abs(odds))


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
