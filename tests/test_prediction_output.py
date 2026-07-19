import numpy as np
import pandas as pd

from tennis_predictor.config import AppConfig
from tennis_predictor.data.names import NameResolver, PlayerIdentity
from tennis_predictor.pipeline import TennisPredictionSystem


class _StubAgeCurve:
    def elo_adjustment(self, age1, age2, scale):
        return np.array([0.0])


class _StubEloModel:
    scale = 400.0

    def predict_proba(self, values):
        return np.array([0.56])


class _StubBoostedModel:
    def predict_proba(self, frame):
        return np.array([0.61])


class _StubCalibrator:
    def predict(self, values):
        return np.array([0.63])


def test_prediction_includes_dashboard_feature_snapshot():
    system = TennisPredictionSystem(AppConfig())
    system.fitted = True
    system.name_resolver = NameResolver(
        [
            PlayerIdentity("ATP:1", "Carlos Alcaraz", "carlos alcaraz", "ATP"),
            PlayerIdentity("ATP:2", "Jannik Sinner", "jannik sinner", "ATP"),
        ]
    )
    system.calendar = None
    system.age_curve = _StubAgeCurve()
    system.elo_model = _StubEloModel()
    system.boosted_model = _StubBoostedModel()
    system.calibrator = _StubCalibrator()
    system.feature_engineer.state.as_of = pd.Timestamp("2026-01-01")

    result = system.predict_match(
        "Carlos Alcaraz",
        "Jannik Sinner",
        "Hard",
        "Masters",
        "2026-07-15",
        tour="ATP",
    )

    assert result["player1_win_probability"] == 0.63
    assert result["feature_snapshot"]["elo_blended1"] == 1500.0
    assert result["feature_snapshot"]["h2h_matches"] == 0
