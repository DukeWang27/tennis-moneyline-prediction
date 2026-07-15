import joblib

from tennis_predictor.config import FeatureConfig, FilterConfig
from tennis_predictor.features.state import TennisState


def test_state_is_serializable(tmp_path):
    state = TennisState(FeatureConfig(), FilterConfig())
    state.h2h[("ATP:1", "ATP:2")]["ATP:1"] = 2
    path = tmp_path / "state.joblib"
    joblib.dump(state, path)
    loaded = joblib.load(path)
    assert loaded.h2h[("ATP:1", "ATP:2")]["ATP:1"] == 2
