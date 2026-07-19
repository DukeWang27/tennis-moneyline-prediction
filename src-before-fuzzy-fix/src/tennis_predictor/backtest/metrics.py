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
    if "bet_side" not in frame:
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
        "mean_clv_probability": placed.get("clv_probability", pd.Series(dtype=float)).mean(),
        "mean_clv_price": placed.get("clv_price", pd.Series(dtype=float)).mean(),
    }


def expected_calibration_error(
    actual: pd.Series | np.ndarray,
    probability: pd.Series | np.ndarray,
    *,
    bins: int = 10,
) -> float:
    y = np.asarray(actual, dtype=float)
    p = np.asarray(probability, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], np.clip(p[mask], 0.0, 1.0)
    if len(y) == 0:
        return np.nan
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    indices = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    ece = 0.0
    for index in range(bins):
        selected = indices == index
        if not selected.any():
            continue
        ece += selected.mean() * abs(float(y[selected].mean()) - float(p[selected].mean()))
    return float(ece)


def _derive_competition_segment(frame: pd.DataFrame) -> pd.Series:
    if "competition_segment" in frame:
        existing = frame["competition_segment"].fillna("").astype(str)
    else:
        existing = pd.Series("", index=frame.index)
    tour = frame.get("tour", pd.Series("", index=frame.index)).fillna("").astype(str).str.upper()
    tier = frame.get("tier", pd.Series("", index=frame.index)).fillna("").astype(str).str.casefold()
    result = existing.where(existing.ne(""), "OTHER")
    result = result.mask(tour.eq("ATP") & tier.eq("challenger"), "ATP_CHALLENGER")
    result = result.mask(tour.eq("ATP") & tier.eq("itf"), "ITF_MEN")
    result = result.mask(tour.eq("WTA") & tier.eq("itf"), "ITF_WOMEN")
    result = result.mask(tour.eq("ATP") & result.eq("OTHER"), "ATP_MAIN")
    result = result.mask(tour.eq("WTA") & result.eq("OTHER"), "WTA_MAIN")
    return result


def calibration_curve_table(predictions: pd.DataFrame, *, bins: int = 10) -> pd.DataFrame:
    if predictions is None or predictions.empty:
        return pd.DataFrame()
    frame = predictions.copy()
    frame["competition_segment"] = _derive_competition_segment(frame)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    rows: list[dict] = []
    for model, model_group in frame.groupby("model", dropna=False):
        views = {"all": model_group}
        if "eligible_with_filters" in model_group:
            views["filtered"] = model_group[model_group["eligible_with_filters"]]
        for view_name, view in views.items():
            groups = [("full", "all", view)]
            groups.extend(
                ("competition_segment", str(segment), group)
                for segment, group in view.groupby("competition_segment", dropna=False)
            )
            for segment_type, segment, group in groups:
                valid = group.dropna(subset=["actual", "probability"]).copy()
                if valid.empty:
                    continue
                valid["bin"] = pd.cut(
                    valid["probability"].clip(0.0, 1.0),
                    bins=edges,
                    labels=False,
                    include_lowest=True,
                    right=True,
                )
                for bin_index, bucket in valid.groupby("bin", dropna=True):
                    if bucket.empty:
                        continue
                    index = int(bin_index)
                    rows.append(
                        {
                            "filter_view": view_name,
                            "model": str(model),
                            "segment_type": segment_type,
                            "segment": segment,
                            "bin": index,
                            "bin_low": float(edges[index]),
                            "bin_high": float(edges[index + 1]),
                            "matches": len(bucket),
                            "mean_probability": float(bucket["probability"].mean()),
                            "actual_win_rate": float(bucket["actual"].mean()),
                            "calibration_gap": float(
                                bucket["actual"].mean() - bucket["probability"].mean()
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def evaluate_predictions(
    predictions: pd.DataFrame,
    *,
    bootstrap_samples: int = 1000,
    random_state: int = 27,
) -> pd.DataFrame:
    rows: list[dict] = []
    frame = predictions.copy()
    frame["competition_segment"] = _derive_competition_segment(frame)
    views = {"all": frame}
    if "eligible_with_filters" in frame:
        views["filtered"] = frame[frame["eligible_with_filters"]]
    for view_name, view in views.items():
        segments: list[tuple[str, str, pd.DataFrame]] = []
        for model, model_group in view.groupby("model", dropna=False):
            segments.append(("full", str(model), model_group))
            for tour, group in model_group.groupby("tour", dropna=False):
                segments.append(("tour", f"{model}|{tour}", group))
            for tier, group in model_group.groupby("tier", dropna=False):
                segments.append(("tier", f"{model}|{tier}", group))
            for segment, group in model_group.groupby("competition_segment", dropna=False):
                segments.append(("competition_segment", f"{model}|{segment}", group))

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
                "ece": expected_calibration_error(y, p),
                "mean_probability": p.mean(),
                "actual_win_rate": y.mean(),
            }
            row.update(_bet_metrics(valid, bootstrap_samples, random_state))
            rows.append(row)
    return pd.DataFrame(rows)
