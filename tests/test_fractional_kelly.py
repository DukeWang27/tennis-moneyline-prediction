import pytest

from tennis_predictor.market import betting_signal, fractional_kelly


def test_quarter_kelly_and_signal():
    rec = fractional_kelly(0.60, 2.0, 100.0, multiplier=0.25, max_fraction=None)
    # Full Kelly = (0.60 * 1 - 0.40) / 1 = 0.20; quarter Kelly = 0.05.
    assert rec.full_kelly_fraction == pytest.approx(0.20)
    assert rec.applied_fraction == pytest.approx(0.05)
    assert rec.stake == pytest.approx(5.00)
    assert rec.edge == pytest.approx(0.10)
    assert betting_signal(rec, min_edge=0.03) == "PAPER-BET CANDIDATE"


def test_negative_kelly_is_a_pass():
    rec = fractional_kelly(0.40, 2.0, 100.0)
    assert rec.applied_fraction == 0
    assert rec.stake == 0
    assert betting_signal(rec) == "PASS"
