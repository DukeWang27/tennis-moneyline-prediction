import pandas as pd

from tennis_predictor.backtest.metrics import evaluate_predictions
from tennis_predictor.services.candidate_training import evaluate_model_gate


def _predictions(probabilities, actual, segment):
    n = len(probabilities)
    return pd.DataFrame({
        "model": ["boosted"] * n,
        "tour": ["ATP"] * n,
        "tier": ["Challenger"] * n,
        "competition_segment": [segment] * n,
        "actual": actual,
        "probability": probabilities,
        "eligible_with_filters": [True] * n,
    })


def test_metrics_include_required_competition_segment_and_ece():
    predictions = _predictions([0.8, 0.7, 0.2, 0.3], [1, 1, 0, 0], "ATP_CHALLENGER")
    metrics = evaluate_predictions(predictions, bootstrap_samples=10)
    row = metrics[
        metrics["segment_type"].eq("competition_segment")
        & metrics["segment"].eq("boosted|ATP_CHALLENGER")
        & metrics["filter_view"].eq("all")
    ].iloc[0]
    assert row["matches"] == 4
    assert row["ece"] >= 0


def test_gate_requires_better_log_loss_and_calibration():
    rows_candidate = []
    rows_production = []
    for segment in ("ATP_CHALLENGER", "ITF_MEN", "ITF_WOMEN"):
        rows_candidate.append({
            "filter_view": "all", "segment_type": "competition_segment",
            "segment": f"boosted|{segment}", "matches": 250,
            "log_loss": 0.55, "ece": 0.04,
        })
        rows_production.append({
            "filter_view": "all", "segment_type": "competition_segment",
            "segment": f"boosted|{segment}", "matches": 250,
            "log_loss": 0.60, "ece": 0.05,
        })
    gate = evaluate_model_gate(pd.DataFrame(rows_candidate), pd.DataFrame(rows_production))
    assert gate.status == "PASS"
