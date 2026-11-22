"""Odds-anchored model: start from the closing price, let player factors move it, bet only when the odds pay enough.

The formula is

    logit(P(player 1 wins)) = a + b * logit(market chance) + sum_k c_k * factor_k

so with every factor at zero it reproduces the market (b near 1), and each factor can only nudge it. A bet is placed on
the side whose expected value, ``P * decimal_odds - 1``, clears the threshold. Stakes follow a quarter of Kelly, capped,
so a run of upsets on short-priced favorites can't empty the bankroll.

Walk-forward by quarter: each quarter is predicted by a formula fitted only on earlier priced matches. Settings marked
PRIMARY below were fixed before any result was seen; everything else is exploratory and should be read as such.

Usage:
    python scripts/odds_anchored_backtest.py \
        --features outputs/backtest/features.parquet \
        --predictions outputs/backtest/predictions.parquet \
        --extended outputs/extended_features.parquet
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from tennis_predictor.features.extended import EXTENDED_DIFF_FEATURES

EPS = 1e-6
PRIMARY_THRESHOLD = 0.02  # PRIMARY: expected value per $1 needed to bet
PRIMARY_C = 0.05  # PRIMARY: L2 strength for the logistic formula (smaller = factors held closer to zero)
FIRST_TEST_QUARTER = "2022-01-01"  # PRIMARY: 2021 is the first training year
KELLY_FRACTION = 0.25
MAX_STAKE = 0.02  # never more than 2% of the bankroll on one match
BASE_FACTORS = [
    "elo_diff", "age_diff", "form10_diff", "form20_diff", "h2h_diff", "rest_days_diff", "tour_matches_diff",
    "injury_incidents_diff", "rank_diff",
]


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def log_loss(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def load(features_path: str, predictions_path: str, extended_path: str) -> pd.DataFrame:
    features = pd.read_parquet(features_path)
    priced = features.dropna(subset=["market_decimal1_close", "market_decimal2_close", "market_p1_close"]).copy()
    void = priced.reindex(columns=["retirement", "walkover", "defaulted"], fill_value=False).astype(float).fillna(0).gt(0).any(axis=1)
    priced = priced[~void]  # retirements and walkovers are refunded: no result to learn from or bet on
    predictions = pd.read_parquet(predictions_path, columns=["match_id", "model", "probability"])
    boosted = predictions[predictions["model"] == "boosted"].drop_duplicates("match_id").set_index("match_id")["probability"]
    priced["model_probability"] = priced["match_id"].map(boosted)
    extended = pd.read_parquet(extended_path)
    priced = priced.merge(extended, on="match_id", how="left")
    priced["date"] = pd.to_datetime(priced["date"])
    priced["quarter"] = priced["date"].dt.to_period("Q").dt.start_time
    priced["market_logit"] = logit(priced["market_p1_close"])
    priced["overround"] = 1 / priced["market_decimal1_close"] + 1 / priced["market_decimal2_close"] - 1
    priced["model_vs_market"] = logit(priced["model_probability"].fillna(priced["market_p1_close"])) - priced["market_logit"]
    # The favourite-longshot bias bends the price at the extremes; this lets the formula bend it back.
    priced["market_logit_curve"] = priced["market_logit"] * np.abs(priced["market_logit"])
    return priced.sort_values(["date", "match_id"]).reset_index(drop=True)


def design(frame: pd.DataFrame, columns: list[str], stats: tuple[pd.Series, pd.Series] | None = None):
    x = frame[columns].astype(float)
    if stats is None:
        stats = (x.mean(), x.std().replace(0, 1).fillna(1))
    z = ((x - stats[0]) / stats[1]).fillna(0.0).clip(-5, 5)  # missing -> average
    return z, stats


def walk_forward(frame: pd.DataFrame, columns: list[str], c: float) -> np.ndarray:
    out = np.full(len(frame), np.nan)
    quarters = sorted(q for q in frame["quarter"].unique() if q >= pd.Timestamp(FIRST_TEST_QUARTER))
    for quarter in quarters:
        train = frame["quarter"] < quarter
        test = (frame["quarter"] == quarter).to_numpy()
        z_train, stats = design(frame[train], columns)
        z_test, _ = design(frame[test], columns, stats)
        # The market logit is not standardised or penalised much: it is the anchor.
        x_train = np.column_stack([frame.loc[train, "market_logit"], z_train])
        x_test = np.column_stack([frame.loc[test, "market_logit"], z_test])
        model = LogisticRegression(C=c, max_iter=2000).fit(x_train, frame.loc[train, "player1_won"].astype(int))
        out[test] = model.predict_proba(x_test)[:, 1]
    return out


def fit_full(frame: pd.DataFrame, columns: list[str], c: float) -> dict[str, float]:
    z, stats = design(frame, columns)
    x = np.column_stack([frame["market_logit"], z])
    model = LogisticRegression(C=c, max_iter=2000).fit(x, frame["player1_won"].astype(int))
    coefficients = {"intercept": float(model.intercept_[0]), "market_logit": float(model.coef_[0][0])}
    # Report per raw unit as well as per standard deviation so the formula can be applied by hand.
    for name, coef, sd in zip(columns, model.coef_[0][1:], stats[1]):
        coefficients[name] = float(coef)
        coefficients[f"{name} (per unit)"] = float(coef / sd)
    return coefficients


def bets(frame: pd.DataFrame, probability: np.ndarray, threshold: float) -> pd.DataFrame:
    p = probability
    ev1 = p * frame["market_decimal1_close"].to_numpy() - 1
    ev2 = (1 - p) * frame["market_decimal2_close"].to_numpy() - 1
    side1 = (ev1 >= ev2) & (ev1 >= threshold)
    side2 = (ev2 > ev1) & (ev2 >= threshold)
    placed = (side1 | side2) & np.isfinite(p)
    odds = np.where(side1, frame["market_decimal1_close"], frame["market_decimal2_close"])
    chance = np.where(side1, p, 1 - p)
    won = np.where(side1, frame["player1_won"] == 1, frame["player1_won"] == 0)
    kelly = np.clip(KELLY_FRACTION * (chance * odds - 1) / (odds - 1), 0, MAX_STAKE)
    return pd.DataFrame({
        "date": frame["date"], "odds": odds, "chance": chance, "ev": np.where(side1, ev1, ev2), "won": won,
        "profit": np.where(won, odds - 1, -1.0), "kelly_fraction": kelly,
    })[placed]


def summarise(placed: pd.DataFrame, seed: int = 27) -> dict:
    if placed.empty:
        return {"bets": 0}
    profit = placed["profit"].to_numpy()
    rng = np.random.default_rng(seed)
    boot = [profit[rng.integers(0, len(profit), len(profit))].mean() for _ in range(2000)]
    bankroll, peak, worst = 1.0, 1.0, 0.0
    for stake, won, odds in zip(placed["kelly_fraction"], placed["won"], placed["odds"]):
        bankroll += bankroll * stake * ((odds - 1) if won else -1)
        peak = max(peak, bankroll)
        worst = max(worst, 1 - bankroll / peak)
    return {
        "bets": len(placed),
        "win_rate": float(placed["won"].mean()),
        "break_even_win_rate": float((1 / placed["odds"]).mean()),
        "average_odds": float(placed["odds"].mean()),
        "roi": float(profit.mean()),
        "roi_ci": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "kelly_bankroll_multiple": float(bankroll),
        "kelly_max_drawdown": float(worst),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--features", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--extended", required=True)
    parser.add_argument("--json")
    args = parser.parse_args(argv)
    frame = load(args.features, args.predictions, args.extended)
    test = frame["quarter"] >= pd.Timestamp(FIRST_TEST_QUARTER)
    y = frame.loc[test, "player1_won"].to_numpy(float)
    print(f"{len(frame):,} priced, settled matches; test {test.sum():,} from {FIRST_TEST_QUARTER} "
          f"to {frame['date'].max():%Y-%m-%d}")

    variants = {
        "market only (recalibrated)": ["market_logit_curve", "overround"],
        "market + model's chance": ["market_logit_curve", "overround", "model_vs_market"],
        "market + all factors (PRIMARY)": ["market_logit_curve", "overround", "model_vs_market", *BASE_FACTORS,
                                           *EXTENDED_DIFF_FEATURES],
    }
    result = {"variants": {}}
    print(f"\nlog loss on the test quarters (lower is better); raw closing price: {log_loss(y, frame.loc[test, 'market_p1_close'].to_numpy()):.4f}")
    for name, columns in variants.items():
        probability = walk_forward(frame, columns, PRIMARY_C)
        entry = {"log_loss": log_loss(y, probability[test.to_numpy()]), "bets": {}}
        for threshold in (0.0, 0.02, 0.05):
            placed = bets(frame[test], probability[test.to_numpy()], threshold)
            entry["bets"][f"{threshold:.0%}"] = summarise(placed)
        primary = bets(frame[test], probability[test.to_numpy()], PRIMARY_THRESHOLD)
        entry["by_odds"] = {
            label: summarise(primary[(primary["odds"] >= lo) & (primary["odds"] < hi)])
            for label, lo, hi in (("1.01-1.50", 1.0, 1.5), ("1.50-2.00", 1.5, 2.0), ("2.00-3.00", 2.0, 3.0), ("3.00+", 3.0, 1e9))
        }
        result["variants"][name] = entry
        print(f"\n== {name}: log loss {entry['log_loss']:.4f}")
        for label, s in entry["bets"].items():
            if not s["bets"]:
                print(f"   EV >= {label}: no bets")
                continue
            print(f"   EV >= {label}: {s['bets']:>5} bets  win {s['win_rate']:.1%} (break-even {s['break_even_win_rate']:.1%})  "
                  f"avg odds {s['average_odds']:.2f}  ROI {s['roi']:+.1%} (95% {s['roi_ci'][0]:+.1%} to {s['roi_ci'][1]:+.1%})  "
                  f"quarter-Kelly bankroll x{s['kelly_bankroll_multiple']:.2f}, worst drop {s['kelly_max_drawdown']:.0%}")
        for label, s in entry["by_odds"].items():
            if s.get("bets"):
                print(f"   odds {label}: {s['bets']:>5} bets  win {s['win_rate']:.1%} (break-even {s['break_even_win_rate']:.1%})  ROI {s['roi']:+.1%} "
                      f"(95% {s['roi_ci'][0]:+.1%} to {s['roi_ci'][1]:+.1%})")

    columns = variants["market + all factors (PRIMARY)"]
    coefficients = fit_full(frame, columns, PRIMARY_C)
    result["formula_fitted_on_all_priced_matches"] = coefficients
    print("\nPRIMARY formula fitted on all priced matches (coefficients per standard deviation, largest first):")
    print(f"   market_logit {coefficients['market_logit']:+.3f}   intercept {coefficients['intercept']:+.3f}")
    for name in sorted(columns, key=lambda n: -abs(coefficients[n]))[:15]:
        print(f"   {name:36s} {coefficients[name]:+.3f}")
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(result, handle, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
