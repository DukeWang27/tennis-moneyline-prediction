"""The market benchmark, voided bets, the orientation guard, and the closing-line feature staying out of the model."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tennis_predictor.backtest.engine import WalkForwardBacktester
from tennis_predictor.backtest.metrics import evaluate_predictions
from tennis_predictor.features.engineer import FeatureEngineer


def _predictions(n=400, seed=3):
    rng = np.random.default_rng(seed)
    market = rng.uniform(0.2, 0.8, n)
    actual = (rng.uniform(size=n) < market).astype(int)      # the market is calibrated...
    model = np.clip(market + rng.normal(0, 0.15, n), 0.02, 0.98)   # ...and the model is the market plus noise
    return pd.DataFrame({
        "model": ["boosted"] * n, "tour": ["ATP"] * n, "tier": ["ATP 250"] * n, "competition_segment": ["ATP_MAIN"] * n,
        "actual": actual, "probability": model, "market_p1_close": market, "eligible_with_filters": [True] * n,
    })


def test_market_log_loss_is_reported_next_to_the_models_on_the_same_rows():
    frame = _predictions()
    frame.loc[:49, "market_p1_close"] = np.nan                  # rows without a price are left out of both numbers
    metrics = evaluate_predictions(frame, bootstrap_samples=10)
    row = metrics[(metrics["filter_view"] == "all") & (metrics["segment_type"] == "full")].iloc[0]
    assert row["market_matches"] == 350
    assert row["log_loss_vs_market"] == pytest.approx(row["model_log_loss_same_rows"] - row["market_log_loss"])
    assert row["log_loss_vs_market"] > 0                        # noise on top of a calibrated market can only hurt
    assert row["market_brier"] < row["model_brier_same_rows"]


def test_market_benchmark_is_empty_without_closing_prices():
    frame = _predictions().drop(columns=["market_p1_close"])
    row = evaluate_predictions(frame, bootstrap_samples=10).iloc[0]
    assert row["market_matches"] == 0 and np.isnan(row["market_log_loss"])


def _backtester():
    bt = WalkForwardBacktester.__new__(WalkForwardBacktester)
    bt.backtest_config = SimpleNamespace(edge_threshold=0.02, kelly_fraction=0.25, max_kelly_fraction=0.05)
    return bt


def _bet_frame(**flags):
    base = {
        "probability": [0.70, 0.70, 0.70], "actual": [1, 0, 1],
        "market_p1_open": [0.60, 0.60, 0.60], "market_p1_close": [0.62, 0.62, 0.62],
        "market_decimal1_open": [1.6, 1.6, 1.6], "market_decimal2_open": [2.5, 2.5, 2.5],
        "market_decimal1_close": [1.55, 1.55, 1.55], "market_decimal2_close": [2.6, 2.6, 2.6],
        "retirement": [False, False, False], "walkover": [False, False, False], "defaulted": [False, False, False],
    }
    base.update(flags)
    return pd.DataFrame(base)


def test_retirements_and_walkovers_are_voided_not_settled():
    out = _backtester()._add_bets(_bet_frame(retirement=[False, True, False], walkover=[False, False, True]))
    assert out["bet_side"].tolist() == [1, 0, 0]
    assert out["voided"].tolist() == [False, True, True]
    assert out["flat_profit"].tolist() == pytest.approx([0.6, 0.0, 0.0])
    assert out["kelly_stake"].iloc[1] == 0.0 and out["kelly_profit"].iloc[2] == 0.0


def test_a_completed_match_still_bets_and_a_frame_without_the_flags_still_works():
    out = _backtester()._add_bets(_bet_frame().drop(columns=["retirement", "walkover", "defaulted"]))
    assert out["bet_side"].tolist() == [1, 1, 1] and not out["voided"].any()


def test_winner_oriented_data_is_refused():
    n = 600
    frame = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n, freq="D"), "match_id": range(n),
                          "player1_won": [1] * 480 + [0] * 120})
    with pytest.raises(ValueError, match="oriented by the winner"):
        _backtester().run(frame)


def test_closing_line_move_is_not_a_model_feature():
    assert "large_line_move" not in FeatureEngineer.numeric_feature_columns()
