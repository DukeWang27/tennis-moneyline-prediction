import pytest

from tennis_predictor.market.odds_math import manual_odds_to_decimal


def test_manual_odds_formats_convert_to_decimal():
    assert manual_odds_to_decimal("+150", "American") == pytest.approx(2.5)
    assert manual_odds_to_decimal("-200", "American") == pytest.approx(1.5)
    assert manual_odds_to_decimal("2.10", "Decimal") == pytest.approx(2.10)
    assert manual_odds_to_decimal("0.50", "Polymarket price") == pytest.approx(2.0)
    assert manual_odds_to_decimal("50", "Polymarket price") == pytest.approx(2.0)


def test_invalid_manual_odds_are_rejected():
    with pytest.raises(ValueError):
        manual_odds_to_decimal("-50", "American")
    with pytest.raises(ValueError):
        manual_odds_to_decimal("1.0", "Decimal")
    with pytest.raises(ValueError):
        manual_odds_to_decimal("100", "Polymarket price")
