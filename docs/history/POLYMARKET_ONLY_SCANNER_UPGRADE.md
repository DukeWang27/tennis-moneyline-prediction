# Polymarket-Only Tennis Scanner Upgrade

This overlay modifies the existing Tennis Edge Lab project. It does not replace the LightGBM model, walk-forward backtest, historical features, or API-Tennis result ingestion.

## What changes

- Renames **Today's Matches** to **Polymarket Scanner**.
- Uses Polymarket as the only odds/market source.
- Uses Gamma `/public-search` with tennis-related queries.
- Falls back to Gamma `/markets/keyset` cursor pagination.
- Reads executable asks, bids, spreads, and visible ask liquidity from the public CLOB order book.
- Shows only API-Tennis fixtures that match an active Polymarket tennis winner market.
- Keeps API-Tennis for surface, tier, complete scheduling, completed results, and player-state updates.
- Removes FanDuel/The Odds API from the live-board workflow.
- Keeps manual Polymarket price entry as a fallback.

## Environment

Only `API_TENNIS_KEY` is needed for the live scanner. Polymarket market discovery and read-only CLOB prices require no authentication.

## Tests

After installation, `pytest` should report 15 passed.
