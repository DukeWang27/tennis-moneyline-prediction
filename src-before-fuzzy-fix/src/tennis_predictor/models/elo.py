from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit


class EloLogisticModel:
    """p(player1) = sigmoid(Elo difference / fitted scale)."""

    def __init__(self, scale: float = 173.7178):
        self.scale = float(scale)

    def fit(self, elo_difference, y, sample_weight=None) -> "EloLogisticModel":
        diff = np.asarray(elo_difference, dtype=float)
        target = np.asarray(y, dtype=float)
        weight = np.ones_like(target) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        mask = np.isfinite(diff) & np.isfinite(target) & np.isfinite(weight) & (weight > 0)
        diff, target, weight = diff[mask], target[mask], weight[mask]
        if len(diff) < 20:
            return self

        def objective(log_scale: float) -> float:
            scale = np.exp(log_scale)
            p = np.clip(expit(diff / scale), 1e-8, 1 - 1e-8)
            losses = -(target * np.log(p) + (1 - target) * np.log(1 - p))
            return float(np.average(losses, weights=weight))

        result = minimize_scalar(
            objective,
            bounds=(np.log(30.0), np.log(800.0)),
            method="bounded",
        )
        if result.success:
            self.scale = float(np.exp(result.x))
        return self

    def predict_proba(self, elo_difference) -> np.ndarray:
        return expit(np.asarray(elo_difference, dtype=float) / self.scale)
