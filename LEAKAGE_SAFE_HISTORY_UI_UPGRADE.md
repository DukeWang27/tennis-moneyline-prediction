# Scanner display + leakage-safe historical replay

This overlay keeps the existing model, feature engine, incremental state updates, and walk-forward backtester unchanged.

## Scanner display

- Raw API-Tennis nested payloads are never rendered.
- Unresolved fixtures are summarized by exclusion reason.
- The audit table shows only date, time, matchup, tournament, tour, tier, status, and reason.
- Polymarket order-book data is displayed as a two-row outcome table with cents, spread, decimal odds, visible liquidity, and match confidence.

## Historical replay

The Match Predictor now has two modes:

1. **Current / upcoming** — uses the currently fitted state and rejects dates before that state.
2. **Historical replay — leakage safe** — reads the exact saved probabilities from `outputs/backtest/predictions.parquet`.

Historical replay never invokes the current fitted model. Only matches that were out-of-sample test rows in the chronological walk-forward backtest are selectable. The actual result is hidden by default.

This intentionally does not create retroactive predictions for training-era matches or for recent matches that were never predicted before they occurred.
