# ITF sample-quality gate + bounded Polymarket scanner

This overlay does not retrain or change the LightGBM model, Elo formulas, or historical feature definitions.

## Sample-quality rules

- **INSUFFICIENT**: either player has fewer than 8 recorded matches. No prediction or paper bet is shown.
- **LIMITED**: prediction may be shown for research, but paper betting is blocked when either player has fewer than 20 total matches, fewer than 5 matches on the selected surface, or data older than 120 days.
- **SUFFICIENT**: both players clear all thresholds. Only these matchups can receive a PAPER-BET CANDIDATE signal.

The predictor and scanner show total matches, surface matches, and last recorded match date for both players. The scanner also lists better-covered alternatives.

## Scanner stability rules

- The scanner no longer refreshes automatically when the page opens.
- Gamma discovery is bounded to one targeted search page per query and at most four keyset fallback pages.
- Markets are filtered to the selected date before CLOB order books are requested.
- CLOB books use `POST /books` batches, with individual GET fallback.
- At most 60 markets are retained for a day and only 10 cards are rendered per browser page.
- Large nested Gamma payloads are not stored in Streamlit session state.
- If a refresh fails, the last successful board stays visible.
