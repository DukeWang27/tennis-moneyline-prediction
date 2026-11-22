"""Price a binary contract against a sharp bookmaker's line.

The walk-forward backtest (see README, "Results") found the model adds nothing to Pinnacle's closing price, while
buying other books' prices only when they beat Pinnacle's de-vigged price was the one approach that was not clearly
negative. This module is that comparison for exchange-style contracts (Kalshi, Polymarket US): a contract that pays
$1 if the player wins, bought at ``price`` dollars plus a fee.

Nothing here places orders. It turns two prices into an expected value so a scanner or a paper log can use it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .odds_math import devig_two_way

# Kalshi's published taker fee is ceil(rate * contracts * price * (1 - price)) dollars, rounded up to the cent, with
# rate 0.07 for most markets. Check the current fee schedule before relying on it; some series use other rates and
# resting (maker) orders are charged differently.
KALSHI_TAKER_FEE_RATE = 0.07


def kalshi_taker_fee(price: float, contracts: int, rate: float = KALSHI_TAKER_FEE_RATE) -> float:
    """Total fee in dollars for buying ``contracts`` at ``price`` (0-1) as a taker."""
    if not 0.0 < price < 1.0 or contracts <= 0:
        raise ValueError("price must be strictly between 0 and 1 and contracts positive")
    raw = rate * contracts * price * (1.0 - price)
    return math.ceil(round(raw * 100, 9)) / 100.0


@dataclass(frozen=True, slots=True)
class ContractValue:
    fair_probability: float  # from the sharp line, vig removed
    price: float  # dollars per contract on the exchange
    fee_per_contract: float
    cost_per_contract: float  # price + fee
    expected_profit_per_contract: float  # fair_probability * $1 - cost
    expected_return: float  # expected profit / cost

    @property
    def is_value(self) -> bool:
        return self.expected_profit_per_contract > 0


def contract_value(
    sharp_decimal_for: float,
    sharp_decimal_against: float,
    price: float,
    *,
    contracts: int = 10,
    fee_rate: float = KALSHI_TAKER_FEE_RATE,
) -> ContractValue:
    """Value of buying the "yes" contract on a player at ``price`` when a sharp book offers the given decimal odds.

    ``sharp_decimal_for`` is the sharp book's price on the same player and ``sharp_decimal_against`` on the opponent.
    The fee is spread over ``contracts`` because Kalshi rounds the whole order's fee up to the cent.
    """
    fair, _ = devig_two_way(sharp_decimal_for, sharp_decimal_against, fmt="decimal")
    fee = kalshi_taker_fee(price, contracts, fee_rate) / contracts
    cost = price + fee
    profit = fair - cost
    return ContractValue(
        fair_probability=float(fair),
        price=float(price),
        fee_per_contract=fee,
        cost_per_contract=cost,
        expected_profit_per_contract=float(profit),
        expected_return=float(profit / cost),
    )


def max_price_for_edge(
    sharp_decimal_for: float,
    sharp_decimal_against: float,
    min_edge: float = 0.02,
    *,
    contracts: int = 10,
    fee_rate: float = KALSHI_TAKER_FEE_RATE,
) -> float | None:
    """Highest whole-cent price at which buying still clears ``min_edge`` expected profit per contract, after fees.

    Useful as the limit price for a resting order. Returns None if no price from 1 to 99 cents qualifies.
    """
    for cents in range(99, 0, -1):
        value = contract_value(
            sharp_decimal_for, sharp_decimal_against, cents / 100, contracts=contracts, fee_rate=fee_rate
        )
        if value.expected_profit_per_contract >= min_edge:
            return cents / 100
    return None
