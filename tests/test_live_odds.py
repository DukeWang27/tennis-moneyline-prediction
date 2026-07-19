import pandas as pd
import pytest

from tennis_predictor.market import BestPriceEngine


def test_best_price_and_sharp_devig():
    predictions = pd.DataFrame([{"fixture_id": "f1", "p_model1": 0.60}])
    quotes = pd.DataFrame(
        [
            {"fixture_id": "f1", "bookmaker": "DraftKings", "bookmaker_key": "draftkings", "decimal_odds1": 2.00, "decimal_odds2": 1.80},
            {"fixture_id": "f1", "bookmaker": "Pinnacle", "bookmaker_key": "pinnacle", "decimal_odds1": 1.95, "decimal_odds2": 1.95},
            {"fixture_id": "f1", "bookmaker": "Bet365", "bookmaker_key": "bet365", "decimal_odds1": 2.10, "decimal_odds2": 1.75},
        ]
    )
    result = BestPriceEngine().summarize(quotes, predictions).iloc[0]
    assert result.best_odds1 == 2.10
    assert result.best_book1 == "Bet365"
    assert result.best_odds2 == 1.95
    assert result.best_book2 == "Pinnacle"
    assert result.sharp_book == "Pinnacle"
    assert result.fair_market_p1 == pytest.approx(0.5)
    assert result.edge1 == pytest.approx(0.10)
    assert result.edge2 == pytest.approx(-0.10)
