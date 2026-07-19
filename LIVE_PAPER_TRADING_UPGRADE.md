# Live data, best-price scanning, and paper trading

This upgrade extends the existing project. It does not replace or retrain the LightGBM model during daily synchronization.

## New environment variables

```bash
export API_TENNIS_KEY="your-api-tennis-key"
export ODDS_API_KEY="your-the-odds-api-key"
```

Polymarket market discovery and read-only CLOB prices use public endpoints and do not require a key.

## One-time database initialization

The database is initialized automatically by the Streamlit app and sync command. To initialize it directly:

```bash
python -c "from tennis_predictor.storage import initialize_paper_trading_db; initialize_paper_trading_db('data/paper_trading.db')"
```

The first wallet row is exactly `$100.00`.

## First post-June 7 backfill

```bash
source .venv/bin/activate
python -m pip install -e ".[dev]"
tennis-sync \
  --config config.yaml \
  --model outputs/backtest/trained_system.joblib \
  --start 2026-06-08
```

The command:

1. Fetches completed API-Tennis singles fixtures.
2. Resolves player identities and tournament metadata.
3. Appends unseen `api_tennis:<event_key>` rows to `data/processed/matches.parquet`.
4. Passes those rows through the existing `FeatureEngineer.transform(..., reset=False)` path.
5. Saves the updated player state back into `trained_system.joblib` without retraining LightGBM.
6. Settles any matching pending paper bets.

Unknown surfaces are written to `data/live/unresolved_fixtures.csv` and are not guessed.

## Daily synchronization

Run this once per day:

```bash
tennis-sync --config config.yaml --model outputs/backtest/trained_system.joblib
```

On macOS, create `~/Library/LaunchAgents/com.tennis-edge.sync.plist` and point the program arguments at the absolute paths for your virtual-environment `tennis-sync`, `config.yaml`, and model file. Load it with `launchctl bootstrap gui/$(id -u) ...`. Keep API keys in a protected environment file or secret manager rather than committing them.

## Streamlit

```bash
python -m streamlit run streamlit_app.py
```

New UI areas:

- Global bankroll / record / ROI banner
- **Today's Matches** page
- Multi-book odds comparison and best-price labels
- Model-versus-de-vigged-market edge
- Paper-bet placement
- **Paper-Bet Tracker** with Pending and Settled tabs

## Surface handling

Surface resolution is intentionally conservative:

1. Use a provider surface field when present.
2. Use `data/metadata/tournaments.csv` by provider tournament key.
3. Fall back to the dominant surface/tier for the same tournament name in the existing historical parquet.
4. Write unresolved events for review and withhold prediction.

To resolve a new tournament, add or edit a row in `data/metadata/tournaments.csv` with `surface` equal to `Hard`, `Clay`, `Grass`, or `Carpet`.

## Important limitations

- The Odds API does not necessarily cover every API-Tennis fixture.
- Polymarket market titles can vary; low-confidence name matches are left unaligned.
- A live-board row can still appear without odds.
- Polymarket decimal odds are derived from the best available ask when an order book is available, otherwise the displayed outcome price is used as a fallback.
- Daily synchronization updates player state. Full LightGBM retraining and walk-forward validation remain separate research operations.
