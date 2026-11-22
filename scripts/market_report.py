"""Does the model know anything the closing price doesn't?

Reads a backtest's predictions.parquet and, on matches with a closing price, compares:

* log loss of the de-vigged closing price, of each model, and of a walk-forward blend
  ``logit(p) = a + b * logit(market) + c * logit(model)`` fitted only on earlier folds;
* the blend's weight on the model (``c``) with a 95% interval. If the interval includes zero, the model adds nothing
  the price doesn't already have, and no betting rule built on it can be trusted;
* flat-stake results of betting the model's edge at the closing price, with a bootstrap interval.

Usage:
    python scripts/market_report.py outputs/backtest/predictions.parquet [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _fit_logistic(x: np.ndarray, y: np.ndarray, iterations: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Unpenalised logistic regression by Newton's method. Returns coefficients and their standard errors."""
    x = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(x.shape[1])
    for _ in range(iterations):
        p = 1 / (1 + np.exp(-x @ beta))
        w = p * (1 - p)
        hessian = x.T @ (x * w[:, None])
        step = np.linalg.solve(hessian, x.T @ (y - p))
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = 1 / (1 + np.exp(-x @ beta))
    covariance = np.linalg.inv(x.T @ (x * (p * (1 - p))[:, None]))
    return beta, np.sqrt(np.diag(covariance))


def _bootstrap_roi(profit: np.ndarray, samples: int = 2000, seed: int = 27) -> tuple[float, float]:
    if len(profit) < 20:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [profit[rng.integers(0, len(profit), len(profit))].mean() for _ in range(samples)]
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def walk_forward_blend(frame: pd.DataFrame) -> np.ndarray:
    """Blend probabilities for each fold, fitted only on the folds before it (NaN for the first fold)."""
    output = np.full(len(frame), np.nan)
    folds = sorted(frame["fold"].unique())
    for fold in folds[1:]:
        train = frame["fold"] < fold
        test = (frame["fold"] == fold).to_numpy()
        if train.sum() < 300:
            continue
        x_train = np.column_stack([_logit(frame.loc[train, "market"]), _logit(frame.loc[train, "probability"])])
        beta, _ = _fit_logistic(x_train, frame.loc[train, "actual"].to_numpy(dtype=float))
        x_test = np.column_stack([np.ones(test.sum()), _logit(frame.loc[test, "market"]), _logit(frame.loc[test, "probability"])])
        output[test] = 1 / (1 + np.exp(-x_test @ beta))
    return output


def bet_results(frame: pd.DataFrame, probability: str, threshold: float) -> dict:
    p = frame[probability].to_numpy()
    market = frame["market"].to_numpy()
    side1 = p - market >= threshold
    side2 = market - p >= threshold
    void = frame.reindex(columns=["retirement", "walkover", "defaulted"], fill_value=False).astype(float).fillna(0).gt(0).any(axis=1).to_numpy()
    placed = (side1 | side2) & ~void
    odds = np.where(side1, frame["market_decimal1_close"], frame["market_decimal2_close"])
    won = np.where(side1, frame["actual"] == 1, frame["actual"] == 0)
    profit = np.where(won, odds - 1, -1.0)[placed]
    low, high = _bootstrap_roi(profit)
    return {"threshold": threshold, "bets": int(placed.sum()), "roi": float(profit.mean()) if len(profit) else float("nan"), "roi_ci_low": low, "roi_ci_high": high}


def report(predictions: pd.DataFrame) -> dict:
    frame = predictions.dropna(subset=["market_p1_close", "probability", "actual"]).copy()
    frame = frame.rename(columns={"market_p1_close": "market"})
    result: dict = {"models": {}}
    for model, group in frame.groupby("model"):
        group = group.sort_values(["fold", "date"]).reset_index(drop=True)
        y = group["actual"].to_numpy(dtype=float)
        x = np.column_stack([_logit(group["market"]), _logit(group["probability"])])
        beta, se = _fit_logistic(x, y)
        group["blend"] = walk_forward_blend(group)
        scored = group.dropna(subset=["blend"])
        entry = {
            "matches_with_price": len(group),
            "log_loss_market": _log_loss(y, group["market"].to_numpy()),
            "log_loss_model": _log_loss(y, group["probability"].to_numpy()),
            "model_weight_in_blend": float(beta[2]),
            "model_weight_ci": [float(beta[2] - 1.96 * se[2]), float(beta[2] + 1.96 * se[2])],
            "market_weight_in_blend": float(beta[1]),
            "walk_forward_blend": {
                "matches": len(scored),
                "log_loss_market": _log_loss(scored["actual"].to_numpy(float), scored["market"].to_numpy()),
                "log_loss_blend": _log_loss(scored["actual"].to_numpy(float), scored["blend"].to_numpy()),
            },
            "bets_model_vs_close": [bet_results(group, "probability", t) for t in (0.02, 0.05, 0.10)],
            "bets_blend_vs_close": [bet_results(scored, "blend", t) for t in (0.01, 0.02, 0.03)],
            "by_tour": {},
        }
        for tour, sub in group.groupby("tour"):
            ys = sub["actual"].to_numpy(float)
            entry["by_tour"][str(tour)] = {
                "matches": len(sub),
                "log_loss_market": _log_loss(ys, sub["market"].to_numpy()),
                "log_loss_model": _log_loss(ys, sub["probability"].to_numpy()),
            }
        result["models"][str(model)] = entry
    return result


def _print(result: dict) -> None:
    for model, entry in result["models"].items():
        print(f"\n== {model}  ({entry['matches_with_price']:,} matches with a closing price)")
        print(f"  log loss   market {entry['log_loss_market']:.4f}   model {entry['log_loss_model']:.4f}   "
              f"(model minus market {entry['log_loss_model'] - entry['log_loss_market']:+.4f}; below zero = model better)")
        low, high = entry["model_weight_ci"]
        print(f"  blend weight on model {entry['model_weight_in_blend']:+.3f}  (95% interval {low:+.3f} to {high:+.3f}); "
              f"on market {entry['market_weight_in_blend']:+.3f}")
        wf = entry["walk_forward_blend"]
        print(f"  walk-forward blend on {wf['matches']:,} matches: market {wf['log_loss_market']:.4f}  blend {wf['log_loss_blend']:.4f}")
        for label, rows in (("model edge", entry["bets_model_vs_close"]), ("blend edge", entry["bets_blend_vs_close"])):
            for row in rows:
                print(f"  bet {label} >= {row['threshold']:.0%} at the close: {row['bets']:>5} bets  ROI {row['roi']:+.1%}  "
                      f"(95% {row['roi_ci_low']:+.1%} to {row['roi_ci_high']:+.1%})")
        for tour, sub in entry["by_tour"].items():
            print(f"  {tour}: {sub['matches']:,} matches  market {sub['log_loss_market']:.4f}  model {sub['log_loss_model']:.4f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("predictions")
    parser.add_argument("--json")
    args = parser.parse_args(argv)
    result = report(pd.read_parquet(args.predictions))
    _print(result)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(result, handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
