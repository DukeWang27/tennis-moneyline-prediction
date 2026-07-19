import numpy as np
import pandas as pd

from tennis_predictor.services import evaluate_paired_model_gate


def _predictions(probabilities):
    actual = np.tile([0, 1], len(probabilities) // 2)
    return pd.DataFrame(
        {
            "match_fingerprint": [f"m{i}" for i in range(len(probabilities))],
            "model": "boosted",
            "competition_segment": "ITF_MEN",
            "actual": actual,
            "probability": probabilities,
        }
    )


def test_paired_gate_requires_significant_improvement():
    n = 400
    actual = np.tile([0, 1], n // 2)
    candidate = np.where(actual == 1, 0.80, 0.20)
    production = np.where(actual == 1, 0.55, 0.45)
    gate = evaluate_paired_model_gate(
        _predictions(candidate),
        _predictions(production),
        required_segments=("ITF_MEN",),
        minimum_segment_matches=200,
        bootstrap_samples=300,
        confidence=0.95,
    )
    assert gate.status == "PASS"
    assert gate.passed_segments == ["ITF_MEN"]


def test_paired_gate_is_inconclusive_without_overlap():
    candidate = _predictions(np.array([0.2, 0.8] * 10))
    production = _predictions(np.array([0.4, 0.6] * 10))
    production["match_fingerprint"] = "x" + production["match_fingerprint"]
    gate = evaluate_paired_model_gate(
        candidate,
        production,
        required_segments=("ITF_MEN",),
        minimum_segment_matches=10,
        bootstrap_samples=20,
    )
    assert gate.status == "INCONCLUSIVE"
