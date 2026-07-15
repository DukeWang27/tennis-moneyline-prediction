from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss


def bootstrap_roi_ci(
    profit: np.ndarray,
    stake: np.ndarray,
    *,
    samples: int = 1000,
    random_state: int = 27,
) -> tuple[float, float]:
    profit = np.asarray(profit, dtype=float)
    stake = np.asarray(stake, dtype=float)
    mask = np.isfinite(profit) & np.isfinite(stake) & (stake > 0)
    profit, stake = profit[mask], stake[mask]
    if len(profit) < 10:
        return np.nan, np.nan
    rng = np.random.default_rng(random_state)
    values = np.empty(samples)
    for i in range(samples):
        idx = rng.integers(0, len(profit), size=len(profit))
        values[i] = profit[idx].sum() / stake[idx].sum()
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _bet_metrics(frame: pd.DataFrame, bootstrap_samples: int, random_state: int) -> dict[str, float]:
    placed = frame[frame["bet_side"].isin([1, 2])].copy()
    if placed.empty:
        return {
            "bets": 0,
            "flat_roi": np.nan,
            "flat_roi_ci_low": np.nan,
            "flat_roi_ci_high": np.nan,
            "kelly_roi": np.nan,
            "kelly_roi_ci_low": np.nan,
            "kelly_roi_ci_high": np.nan,
            "mean_clv_probability": np.nan,
            "mean_clv_price": np.nan,
        }
    flat_roi = placed["flat_profit"].sum() / placed["flat_stake"].sum()
    flat_low, flat_high = bootstrap_roi_ci(
        placed["flat_profit"].to_numpy(),
        placed["flat_stake"].to_numpy(),
        samples=bootstrap_samples,
        random_state=random_state,
    )
    kelly_mask = placed["kelly_stake"] > 0
    if kelly_mask.any():
        kelly_roi = placed.loc[kelly_mask, "kelly_profit"].sum() / placed.loc[kelly_mask, "kelly_stake"].sum()
        kelly_low, kelly_high = bootstrap_roi_ci(
            placed.loc[kelly_mask, "kelly_profit"].to_numpy(),
            placed.loc[kelly_mask, "kelly_stake"].to_numpy(),
            samples=bootstrap_samples,
            random_state=random_state,
        )
    else:
        kelly_roi = kelly_low = kelly_high = np.nan
    return {
        "bets": len(placed),
        "flat_roi": flat_roi,
        "flat_roi_ci_low": flat_low,
        "flat_roi_ci_high": flat_high,
        "kelly_roi": kelly_roi,
        "kelly_roi_ci_low": kelly_low,
        "kelly_roi_ci_high": kelly_high,
        "mean_clv_probability": placed["clv_probability"].mean(),
        "mean_clv_price": placed["clv_price"].mean(),
    }


def evaluate_predictions(
    predictions: pd.DataFrame,
    *,
    bootstrap_samples: int = 1000,
    random_state: int = 27,
) -> pd.DataFrame:
    rows: list[dict] = []
    views = {
        "all": predictions,
        "filtered": predictions[predictions["eligible_with_filters"]],
    }
    for view_name, view in views.items():
        segments: list[tuple[str, str, pd.DataFrame]] = []
        # Every metric row is model-specific; mixing two probability rows for the
        # same match would make accuracy/calibration numbers uninterpretable.
        for model, model_group in view.groupby("model", dropna=False):
            segments.append(("full", str(model), model_group))
            for tour, group in model_group.groupby("tour", dropna=False):
                segments.append(("tour", f"{model}|{tour}", group))
            for tier, group in model_group.groupby("tier", dropna=False):
                segments.append(("tier", f"{model}|{tier}", group))

        for segment_type, segment_value, group in segments:
            valid = group.dropna(subset=["actual", "probability"])
            if valid.empty:
                continue
            y = valid["actual"].astype(int)
            p = valid["probability"].clip(1e-6, 1 - 1e-6)
            row = {
                "filter_view": view_name,
                "segment_type": segment_type,
                "segment": segment_value,
                "matches": len(valid),
                "accuracy": accuracy_score(y, p >= 0.5),
                "brier": brier_score_loss(y, p),
                "log_loss": log_loss(y, p, labels=[0, 1]),
                "mean_probability": p.mean(),
                "actual_win_rate": y.mean(),
            }
            row.update(_bet_metrics(valid, bootstrap_samples, random_state))
            rows.append(row)
    return pd.DataFrame(rows)
