# Major Live Pipeline Overhaul

This overlay refactors the existing project in place. It does not replace the LightGBM model, Elo mathematics, feature definitions, or walk-forward backtester.

## Actual project mapping

- Requested `data_pipeline.py` functionality lives in:
  - `src/tennis_predictor/data/api_tennis.py`
  - `src/tennis_predictor/features/state.py`
  - `src/tennis_predictor/pipeline.py`
  - `src/tennis_predictor/services/live_data.py`
- Requested `db_manager.py` functionality lives in:
  - `src/tennis_predictor/storage/paper_trading.py`
  - `src/tennis_predictor/storage/live_store.py`
- Requested `betting_math.py` is:
  - `src/tennis_predictor/market/betting_math.py`
- Requested `app.py` functionality lives in:
  - `streamlit_app.py`
  - `src/tennis_predictor/ui/pages.py`
  - `src/tennis_predictor/ui/theme.py`

## Important live-state safety rule

The current local model has already been advanced beyond the original June 7, 2026 cutoff. Previously skipped unknown-surface matches from June/July cannot be inserted into that already-advanced state without applying future state to earlier matches.

Use `tennis-rebuild-live` once with a frozen pre-live model artifact:

```bash
BASE_MODEL="$HOME/Downloads/tennis-moneyline-prediction-before-live-upgrade/outputs/backtest/trained_system.joblib"

tennis-rebuild-live \
  --config config.yaml \
  --base-model "$BASE_MODEL" \
  --live-model outputs/backtest/trained_system.joblib \
  --start 2026-06-08 \
  --end 2026-07-15
```

Do not use the current July live model as `--base-model`. If the backup artifact is missing, restore another model saved before the first live sync or rerun the original backtest through June 7.

## Normal daily use after rebuilding

```bash
tennis-sync \
  --config config.yaml \
  --model outputs/backtest/trained_system.joblib
```

Then:

```bash
python -m streamlit run streamlit_app.py
```

## Behavior changes

- Completed matches with unknown surfaces update overall Elo, recent form, rest, H2H, and experience.
- Unknown-surface matches do not update any surface-specific Elo.
- Polymarket is the source of truth for live market discovery; every detected tennis market remains visible.
- Unmapped players can be mapped from the market card and saved in SQLite.
- Unknown tournament surfaces can be selected once and saved in SQLite.
- Low-history players remain visible but receive `COLD START / NO HISTORY`, not a fabricated confident prediction.
- Edge and quarter-Kelly sizing use the executable CLOB best ask.
- Persistent wallet and paper bets are stored in `data/paper_trading.db`.
- Startup settlement checks recent API-Tennis results but does not block the app if the provider is unavailable.
- Historical replay dropdowns query the local match-history store. Probabilities are shown only when a pre-match walk-forward or sequential prediction was persisted.
