import pytest

from tennis_predictor.market.value import (
    contract_value,
    kalshi_taker_fee,
    max_price_for_edge,
)


def test_kalshi_fee_rounds_the_order_up_to_the_cent():
    assert kalshi_taker_fee(0.50, 1) == 0.02  # 0.07 * 0.25 = 0.0175 -> 2 cents
    assert kalshi_taker_fee(0.50, 100) == 1.75
    assert kalshi_taker_fee(0.90, 10) == 0.07  # 0.07 * 10 * 0.09 = 0.063 -> 7 cents
    with pytest.raises(ValueError):
        kalshi_taker_fee(1.0, 1)


def test_contract_value_uses_devigged_sharp_price_and_fee():
    # Pinnacle 1.80 / 2.10: fair = (1/1.8) / (1/1.8 + 1/2.1) = 0.5385
    value = contract_value(1.80, 2.10, 0.50, contracts=100)
    assert value.fair_probability == pytest.approx(0.53846, abs=1e-4)
    assert value.fee_per_contract == pytest.approx(0.0175)
    assert value.expected_profit_per_contract == pytest.approx(0.53846 - 0.5175, abs=1e-4)
    assert value.is_value
    assert not contract_value(1.80, 2.10, 0.53, contracts=100).is_value


def test_max_price_for_edge():
    limit = max_price_for_edge(1.80, 2.10, min_edge=0.02, contracts=100)
    assert limit == 0.50
    assert contract_value(1.80, 2.10, limit, contracts=100).expected_profit_per_contract >= 0.02
    assert contract_value(1.80, 2.10, limit + 0.01, contracts=100).expected_profit_per_contract < 0.02
    assert max_price_for_edge(1.01, 30.0, min_edge=0.05) is None or max_price_for_edge(1.01, 30.0, min_edge=0.05) < 0.95
