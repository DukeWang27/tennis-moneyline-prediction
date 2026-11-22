# Polymarket and Predictor UI Hotfix

This overlay fixes two user-facing problems without changing the LightGBM model,
feature definitions, Elo formulas, or walk-forward validation.

## Polymarket scanner

- Stops paging `/markets` with large numeric offsets, which caused a 422 error.
- Uses targeted Gamma public search plus `/markets/keyset` cursor pagination.
- Uses `start_date_min` and `start_date_max` for the selected New York day.
- Retains markets already found when one discovery source fails.
- Uses only `offset=0` as a compatibility fallback; it never requests offset 2100.
- Keeps every tennis market visible before local player mapping.

## Match Predictor

- Replaces the broken unstyled matchup output with a clean probability panel.
- States the model's likely winner while clearly separating a prediction from a bet.
- Adds best-ask inputs for both players.
- Calculates executable decimal odds, model edge, quarter-Kelly stake, and either
  `PASS` or `PAPER-BET CANDIDATE`.
- Adds a paper-bet ticket directly below the prediction.
- Persists the generated prediction through Streamlit reruns so entering prices or
  pressing the paper-bet button does not make the result disappear.

## Validation

The complete composite project passes 28 tests and Ruff checks.
