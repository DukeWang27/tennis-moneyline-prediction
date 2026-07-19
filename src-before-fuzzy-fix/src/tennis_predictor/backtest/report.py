from __future__ import annotations

import pandas as pd


def print_summary(metrics: pd.DataFrame) -> None:
    if metrics.empty:
        print("No metrics were generated.")
        return
    columns = [
        "filter_view",
        "segment_type",
        "segment",
        "matches",
        "accuracy",
        "brier",
        "log_loss",
        "ece",
        "bets",
        "flat_roi",
        "flat_roi_ci_low",
        "flat_roi_ci_high",
        "kelly_roi",
        "mean_clv_probability",
    ]
    available = [c for c in columns if c in metrics.columns]
    display = metrics[available].copy()
    for col in display.select_dtypes(include="number").columns:
        if col not in {"matches", "bets"}:
            display[col] = display[col].round(4)
    print("\n=== Walk-forward summary ===")
    print(display.to_string(index=False))
