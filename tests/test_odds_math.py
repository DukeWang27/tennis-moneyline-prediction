import pytest

from tennis_predictor.market.odds_math import american_to_decimal, devig_two_way


def test_american_conversion():
    assert american_to_decimal(+150) == pytest.approx(2.5)
    assert american_to_decimal(-200) == pytest.approx(1.5)


def test_devig_sums_to_one():
    p1, p2 = devig_two_way(1.80, 2.10)
    assert p1 + p2 == pytest.approx(1.0)
    assert 0 < p1 < 1
    assert 0 < p2 < 1
