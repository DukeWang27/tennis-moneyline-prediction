from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KellyRecommendation:
    probability: float
    decimal_odds: float
    edge: float
    full_kelly_fraction: float
    applied_fraction: float
    stake: float


def fractional_kelly(
    p_model: float,
    decimal_odds: float,
    bankroll: float,
    *,
    multiplier: float = 0.25,
    max_fraction: float | None = 0.05,
) -> KellyRecommendation:
    """Return a conservative fractional-Kelly stake recommendation.

    The full Kelly fraction is
        (p * (odds - 1) - (1 - p)) / (odds - 1)
    and is multiplied by ``multiplier``. Negative fractions are clipped to zero.
    ``max_fraction`` optionally caps the recommended bankroll fraction.
    """
    p = float(p_model)
    odds = float(decimal_odds)
    roll = float(bankroll)
    fraction_multiplier = float(multiplier)

    if not math.isfinite(p) or not 0.0 <= p <= 1.0:
        raise ValueError("p_model must be a finite probability between 0 and 1")
    if not math.isfinite(odds) or odds <= 1.0:
        raise ValueError("decimal_odds must be finite and greater than 1")
    if not math.isfinite(roll) or roll < 0.0:
        raise ValueError("bankroll must be finite and non-negative")
    if not math.isfinite(fraction_multiplier) or fraction_multiplier < 0.0:
        raise ValueError("multiplier must be finite and non-negative")
    if max_fraction is not None:
        max_fraction = float(max_fraction)
        if not math.isfinite(max_fraction) or max_fraction < 0.0:
            raise ValueError("max_fraction must be finite and non-negative")

    b = odds - 1.0
    q = 1.0 - p
    full_fraction = (p * b - q) / b
    applied = max(full_fraction * fraction_multiplier, 0.0)
    if max_fraction is not None:
        applied = min(applied, max_fraction)
    stake = round(roll * applied, 2)
    edge = p - (1.0 / odds)
    return KellyRecommendation(
        probability=p,
        decimal_odds=odds,
        edge=edge,
        full_kelly_fraction=full_fraction,
        applied_fraction=applied,
        stake=stake,
    )


def betting_signal(
    recommendation: KellyRecommendation,
    *,
    min_edge: float = 0.03,
) -> str:
    """Classify a model/price pair without overriding data-quality warnings."""
    if recommendation.edge <= 0 or recommendation.applied_fraction <= 0:
        return "PASS"
    if recommendation.edge > float(min_edge):
        return "PAPER-BET CANDIDATE"
    return "PASS"
