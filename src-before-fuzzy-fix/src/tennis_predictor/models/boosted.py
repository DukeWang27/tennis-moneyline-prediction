from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

try:
    from lightgbm import LGBMClassifier
except ImportError:  # optional fallback
    LGBMClassifier = None


class ProbabilityCalibrator:
    def __init__(self, method: str = "isotonic"):
        if method not in {"isotonic", "sigmoid", "none"}:
            raise ValueError("Calibration must be isotonic, sigmoid, or none.")
        self.method = method
        self.model = None

    def fit(self, raw_probability, y, sample_weight=None) -> "ProbabilityCalibrator":
        p = np.clip(np.asarray(raw_probability, dtype=float), 1e-6, 1 - 1e-6)
        target = np.asarray(y, dtype=int)
        weight = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if self.method == "none" or len(p) < 100 or len(np.unique(target)) < 2:
            return self
        if self.method == "isotonic" and len(p) >= 1000:
            self.model = IsotonicRegression(out_of_bounds="clip")
            self.model.fit(p, target, sample_weight=weight)
        else:
            self.model = LogisticRegression(C=1e6, max_iter=1000)
            logit = np.log(p / (1 - p)).reshape(-1, 1)
            self.model.fit(logit, target, sample_weight=weight)
        return self

    def predict(self, raw_probability) -> np.ndarray:
        p = np.clip(np.asarray(raw_probability, dtype=float), 1e-6, 1 - 1e-6)
        if self.model is None:
            return p
        if isinstance(self.model, IsotonicRegression):
            return np.clip(self.model.predict(p), 1e-6, 1 - 1e-6)
        logit = np.log(p / (1 - p)).reshape(-1, 1)
        return np.clip(self.model.predict_proba(logit)[:, 1], 1e-6, 1 - 1e-6)


class BoostedProbabilityModel:
    def __init__(self, params: dict, random_state: int = 27):
        params = dict(params)
        params.setdefault("random_state", random_state)
        if LGBMClassifier is not None:
            self.estimator = LGBMClassifier(**params)
            self.backend = "lightgbm"
        else:
            fallback = {
                "learning_rate": params.get("learning_rate", 0.05),
                "max_iter": params.get("n_estimators", 300),
                "max_leaf_nodes": params.get("num_leaves", 31),
                "max_depth": params.get("max_depth", None),
                "min_samples_leaf": params.get("min_child_samples", 50),
                "l2_regularization": params.get("reg_lambda", 1.0),
                "random_state": random_state,
            }
            self.estimator = HistGradientBoostingClassifier(**fallback)
            self.backend = "sklearn_hist"
        self.feature_columns: list[str] = []

    def fit(self, frame: pd.DataFrame, y, feature_columns: list[str], sample_weight=None) -> "BoostedProbabilityModel":
        self.feature_columns = list(feature_columns)
        x = frame.reindex(columns=self.feature_columns).apply(pd.to_numeric, errors="coerce")
        self.estimator.fit(x, np.asarray(y, dtype=int), sample_weight=sample_weight)
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame.reindex(columns=self.feature_columns).apply(pd.to_numeric, errors="coerce")
        return self.estimator.predict_proba(x)[:, 1]
