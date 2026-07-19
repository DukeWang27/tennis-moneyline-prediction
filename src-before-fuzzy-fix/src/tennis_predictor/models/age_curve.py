from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer


class EmpiricalAgeCurve:
    """Fit an antisymmetric spline age effect conditional on Elo difference."""

    def __init__(self, n_knots: int = 6, degree: int = 3, random_state: int = 27):
        self.spline = SplineTransformer(n_knots=n_knots, degree=degree, include_bias=False)
        self.model = LogisticRegression(C=2.0, max_iter=2000, random_state=random_state)
        self.fitted = False
        self.age_coef_: np.ndarray | None = None

    def fit(self, age1, age2, elo_diff, y, sample_weight=None) -> "EmpiricalAgeCurve":
        a1 = np.asarray(age1, dtype=float)
        a2 = np.asarray(age2, dtype=float)
        elo = np.asarray(elo_diff, dtype=float)
        target = np.asarray(y, dtype=int)
        weight = np.ones_like(target, dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        mask = np.isfinite(a1) & np.isfinite(a2) & np.isfinite(elo) & np.isfinite(weight) & (weight > 0)
        if mask.sum() < 500 or len(np.unique(target[mask])) < 2:
            return self
        ages = np.concatenate([a1[mask], a2[mask]])[:, None]
        self.spline.fit(ages)
        b1 = self.spline.transform(a1[mask, None])
        b2 = self.spline.transform(a2[mask, None])
        design = np.column_stack([elo[mask], b1 - b2])
        self.model.fit(design, target[mask], sample_weight=weight[mask])
        self.age_coef_ = self.model.coef_[0, 1:].copy()
        self.fitted = True
        return self

    def age_logit_adjustment(self, age1, age2) -> np.ndarray:
        a1 = np.asarray(age1, dtype=float)
        a2 = np.asarray(age2, dtype=float)
        output = np.zeros_like(a1, dtype=float)
        if not self.fitted or self.age_coef_ is None:
            return output
        mask = np.isfinite(a1) & np.isfinite(a2)
        if mask.any():
            b1 = self.spline.transform(a1[mask, None])
            b2 = self.spline.transform(a2[mask, None])
            output[mask] = (b1 - b2) @ self.age_coef_
        return output

    def elo_adjustment(self, age1, age2, logistic_scale: float) -> np.ndarray:
        return self.age_logit_adjustment(age1, age2) * float(logistic_scale)
