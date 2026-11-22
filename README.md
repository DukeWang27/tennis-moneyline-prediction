# Tennis Moneyline Prediction

A modular Python pipeline for ATP/WTA match ingestion, leakage-safe feature generation, calibrated win probabilities, market comparison, and walk-forward backtesting.

## Results so far (read this first)

Walk-forward backtest, 2021-01 to 2024-02, on the 14,654 ATP and WTA main-tour matches with a Pinnacle closing price
(Tennis-Data). Ratings were built from 554k matches since 2010, including Challenger and ITF; every test quarter was
trained only on earlier data. Run on 2026-11-22 with `config.example.yaml` (fixed K = 32) plus the odds file and `test_start: 2021-01-01`.

| | Log loss (lower is better) | Weight in a market + model blend | Betting the model's edge (>= 2%) at the close |
|---|---|---|---|
| Pinnacle closing price | **0.5879** | +1.05 | n/a |
| LightGBM model | 0.6104 | -0.01 (95%: -0.08 to +0.06) | 11,337 bets, ROI -5.5% (95%: -7.8% to -3.0%) |
| Elo baseline | 0.6277 | -0.06 (95%: -0.12 to +0.01) | 11,863 bets, ROI -5.3% (95%: -7.3% to -3.2%) |

**The model does not beat the market and adds nothing to it.** Blended with the closing price, its weight is
indistinguishable from zero, and a blend fitted walk-forward scores the same as the price alone (0.5892 vs 0.5891).
The earlier Kaggle runs agree (`docs/history/past_runs/`, ROI -9.7% on 15,390 bets). No betting rule on top of these
probabilities should be expected to make money; don't bet them.

**What showed a hint of edge is price comparison, not prediction.** Buying the best price across books only when it
beat Pinnacle's de-vigged price returned +2.0% on 13,640 bets (95%: -0.4% to +4.4%), and Bet365 alone +2.6% on 2,752
(95%: -3.6% to +8.9%). Neither is proven, and the best price across dozens of international books is not something a
US bettor can actually get. `tennis_predictor.market.value` prices exchange contracts (Kalshi, Polymarket US) against a
sharp line, after Kalshi's fee, for paper-testing that idea where you can actually bet.

**Rating update size.** The example config now uses `elo_k_schedule: match_count`, FiveThirtyEight's rule that a
player's rating moves less as they play more. Same data and folds, all 213,010 test matches:

| | Elo log loss | Model log loss | Model log loss on the 14,654 priced matches |
|---|---|---|---|
| fixed K = 32 | 0.6077 | 0.5808 | 0.6104 |
| match_count | **0.5929** | **0.5739** | 0.6088 (market 0.5879) |

Better in every segment (ATP main, WTA main, Challenger, ITF), and still well short of the closing price.

Reproduce: `python scripts/market_report.py outputs/backtest/predictions.parquet` (any backtest output).

**Player factors on top of the odds (2026-11-22).** `scripts/odds_anchored_backtest.py` starts from the closing price
and lets factors nudge it: the model's chance plus height, handedness, serve and return form, aces, double faults,
break points, minutes and matches in recent days, a deep run the week before, home country, travel on short rest,
qualifier/wildcard/seeded status and ranking points (`features/extended.py`). It bets only when expected value clears
+2% and sizes at a quarter of Kelly, capped at 2%. Walk-forward on 9,757 priced matches, 2022-01 to 2024-02:

| | Log loss | Bets at EV >= +2% | ROI (95% interval) | Quarter-Kelly worst drop |
|---|---|---|---|---|
| Closing price as is | 0.5853 | | | |
| Price, recalibrated | 0.5847 | 727 | +6.1% (-2.4% to +14.5%) | 14% |
| Price + model's chance | 0.5848 | 927 | +4.7% (-2.8% to +12.3%) | 14% |
| Price + all factors (pre-registered) | 0.5863 | 4,765 | +1.8% (-1.5% to +5.0%) | 60% |

None is distinguishable from zero, and adding the factors made the probabilities worse than the price alone. A first
run showed +3.8% (+0.9% to +6.9%) for the last row; that came from a leak, now fixed (`features/chronology.py`): exact
odds dates moved some matches after their own next-round match, so "still in the draw tomorrow" leaked the result.
Bets at odds 2.00-3.00 were positive in every row (+5.7% to +8.1%), but that is one slice of many, found after the
fact; treat it as an idea to paper-test, not a result.

## What is implemented

- ATP and WTA yearly Sackmann-format CSV ingestion from attributable GitHub mirrors, including optional ATP qualifying/Challenger and WTA qualifying/ITF files.
- Stable player IDs as the primary identity key, with conservative fuzzy matching for external odds feeds.
- Explicit retirement, walkover, and incomplete-match fields. Walkovers receive zero Elo/model weight by default; retirements receive reduced weight rather than being silently treated like ordinary completed losses.
- Toggleable match flags: round-robin/dead-rubber context, already-qualified context, fatigue, and low-tier event immediately before a Grand Slam.
- Strictly pre-match, streaming features: overall and surface-blended Elo, age, experience, retirement/walkover history, recent form, head-to-head, and rest.
- An empirical spline age curve fitted inside each historical training fold, then expressed as an Elo-equivalent adjustment.
- Maximum-likelihood Elo logistic scale fitting.
- LightGBM probability model with a chronological calibration holdout and isotonic or Platt calibration.
- Two-way de-vigging, model/market edge, line movement, flat-bet ROI, fractional-Kelly ROI, CLV, and bootstrap confidence intervals.
- The market benchmark: every metrics row carries the de-vigged closing price's own log loss and Brier on the rows that have a price, the model's on the same rows, and `log_loss_vs_market` (below zero = the model's probabilities beat the market's). Read this before any ROI figure: a model that loses here cannot be made profitable by a betting rule.
- Retirements, walkovers and defaults are voided in the backtest (stake returned), as on Polymarket and most books; `voided` marks them.
- A guard refuses walk-forward data where player 1 wins more than 65% of matches (winner-oriented data leaks the result into every feature).
- Walk-forward reports for the full sample and by ATP/WTA, tier, and filtered/unfiltered views.
- A serializable `TennisPredictionSystem.predict_match(...)` API.

## Important data caveats

1. **Sackmann match dates:** `tourney_date` is generally the event's Monday/start date, not the exact day each match was played. The loader estimates match dates from round and event type and sets `date_is_estimated=True`. For rigorous fatigue, rest, line movement, and odds alignment, provide exact dates in `data.exact_dates_path` or let aligned odds `commence_time` replace the estimate.
2. **Dead rubbers:** Sackmann does not provide round-robin group membership/standings before each match. Exact flags require `data/context/round_robin_context.csv` with `match_id,dead_rubber`.
3. **Already-qualified qualifying players:** this generally requires draw/alternate/lucky-loser metadata not present in Sackmann. Supply `data/context/qualification_context.csv` with `match_id,already_qualified`.
4. **Tournament tiers:** ATP's historical `A` level does not perfectly distinguish 250 from 500. The loader uses a maintained name heuristic and supports a precise override CSV with `tour,tournament,tier`.
5. **Odds:** The Odds API historical endpoint is paid and snapshot-based. The pipeline includes an adapter and a normalized CSV path so you can use The Odds API, SportsDataIO, or another provider without changing the backtester.
6. **License:** Jeff Sackmann's data is CC BY-NC-SA 4.0. Attribution is required and commercial use is restricted. Check the source license before deployment.

## Installation

```bash
cd tennis-moneyline-prediction
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
cp config.example.yaml config.yaml
```

## Quick start with free data

Results come from Sackmann-format CSVs (the default loader uses public mirrors; the original
`JeffSackmann/tennis_atp` repository is no longer reachable) and odds from Tennis-Data CSVs:

```bash
tennis-download --config config.yaml                       # results into data/processed/matches.parquet
python scripts/tennis_data_to_snapshots.py path/to/tennis_data_csvs data/odds/pinnacle_closing.csv
# in config.yaml: odds.normalized_snapshots_path: data/odds/pinnacle_closing.csv
#                 backtest.test_start: "2021-01-01"   (first year you have odds for)
tennis-backtest --config config.yaml --output-dir outputs/backtest
python scripts/market_report.py outputs/backtest/predictions.parquet
```

A full run (2010 onward, lower levels included) takes about 30 minutes and 6 GB of memory.

Tennis-Data lists players as `"Zverev A."`. The aligner first matches those exactly on surname and first initial
(14,651 of 14,655 aligned prices in the run above), and only falls back to fuzzy matching for other formats; fuzzy
scores alone put `"Zverev A."` below the threshold and can't tell Alexander from Mischa.

## Download and preprocess

```bash
tennis-download --config config.yaml
```

The loader first tries yearly raw files, then falls back to shallow Git clones of maintained mirrors if the raw endpoints are unavailable. You can override the mirror locations with `TENNIS_ATP_REPO_URL`, `TENNIS_WTA_REPO_URL`, `TENNIS_ATP_RAW_URL`, and `TENNIS_WTA_RAW_URL`. Missing years are skipped and reported rather than fabricated.

This creates a processed parquet file containing both an event date and an estimated/overridden match date.

## Optional exact-date file

```csv
match_id,date
ATP_2025-001_17,2025-01-03
```

## Optional context files

`data/context/round_robin_context.csv`

```csv
match_id,dead_rubber
ATP_2025-0605_298,1
```

`data/context/qualification_context.csv`

```csv
match_id,already_qualified
WTA_2025-1234_42,1
```

Unknown context is retained in separate `*_known` columns; it is not silently labeled false in analysis.

## Odds input format

The easiest production workflow is to normalize any provider into one row per bookmaker snapshot/outcome pair:

```csv
snapshot_time,commence_time,tour,tournament,player1,player2,bookmaker,decimal_odds1,decimal_odds2,is_closing
2025-07-01T12:00:00Z,2025-07-01T15:00:00Z,ATP,Wimbledon,Player A,Player B,pinnacle,1.80,2.10,0
2025-07-01T14:55:00Z,2025-07-01T15:00:00Z,ATP,Wimbledon,Player A,Player B,pinnacle,1.72,2.22,1
```

Set `odds.normalized_snapshots_path` in the YAML. The aligner uses date, unordered player names, tour, and tournament similarity, and stores its match score for auditing.

## Run a walk-forward backtest

```bash
tennis-backtest --config config.yaml --output-dir outputs/backtest
```

Outputs:

- `features.parquet`: leakage-safe pre-match features.
- `predictions.parquet`: fold/model probabilities and betting results.
- `metrics.csv`: full and segmented metrics.
- `trained_system.joblib`: final fitted prediction system.

The backtester builds every feature before updating player state with that match, and each fold trains only on dates earlier than its test interval.

## Predict a future match

```bash
tennis-predict \
  --model outputs/backtest/trained_system.joblib \
  --player1 "Carlos Alcaraz" \
  --player2 "Jannik Sinner" \
  --surface Grass \
  --tier Slam \
  --date 2026-07-15
```

Python API:

```python
from tennis_predictor.pipeline import TennisPredictionSystem

system = TennisPredictionSystem.load("outputs/backtest/trained_system.joblib")
result = system.predict_match(
    player1="Carlos Alcaraz",
    player2="Jannik Sinner",
    surface="Grass",
    tournament_tier="Slam",
    match_date="2026-07-15",
)
print(result)
```

A requested prediction date must not precede the fitted state date. For a historical as-of prediction, rebuild the system with a cutoff date.

## Modules not yet in the repository

An old `data/` pattern in `.gitignore` kept eight modules out of Git. Four were recovered; `api_tennis`, `kaggle_sync`,
`surface_audit` and `itf_women_sync` still exist only in the original working copy. Without them the backtest,
prediction and dashboard work; the live API-Tennis sync, Kaggle sync and ITF women's sync commands do not, and their
12 test files are skipped (`pytest -rs` lists the reason). Copy the four files into `src/tennis_predictor/data/` and
commit them to restore everything; nothing else needs to change.

## Testing

```bash
pytest
```

The included tests cover odds math, retirement handling, and the central no-leakage invariant: a match cannot influence its own pre-match features.

## Streamlit dashboard

The repository now includes a multipage dashboard for using the trained system outside the terminal.

```bash
python -m pip install -e ".[dev]"
streamlit run streamlit_app.py
```

Streamlit opens the app at `http://localhost:8501`. The dashboard looks for these local artifacts by default:

- `outputs/backtest/trained_system.joblib`
- `outputs/backtest/metrics.csv`
- `outputs/backtest/predictions.parquet`

The pages include:

- **Overview:** headline walk-forward results and model/tier comparisons.
- **Match Predictor:** player selectors, calibrated probabilities, Elo comparison, filter flags, and player feature comparisons.
- **Backtest Lab:** segment controls, calibration curves, performance through time, and the complete metric table.
- **Match Explorer:** searchable prediction rows with confidence and correctness filters plus CSV export.
- **Model Guide:** interpretation and deployment notes.

### Prepare compact deployment artifacts

Raw match data and `outputs/` remain ignored by Git. Create a smaller dashboard bundle instead:

```bash
tennis-prepare-dashboard \
  --source outputs/backtest \
  --destination dashboard_artifacts
```

This copies the trained system and metrics and writes a reduced, compressed prediction-history file. Check file sizes before committing; normal GitHub files must remain below its file-size limit.

### Deploy on Streamlit Community Cloud

1. Commit and push the application code and `dashboard_artifacts/`.
2. In Streamlit Community Cloud, create an app from the GitHub repository.
3. Set the entrypoint to `streamlit_app.py`.
4. Select Python 3.11 or newer in advanced settings.

`requirements.txt` installs the local project and its Python dependencies. `packages.txt` installs the Linux OpenMP runtime needed by LightGBM.

### Dashboard path overrides

For another artifact location, use the sidebar path controls or environment variables:

```bash
export TENNIS_MODEL_PATH=/path/to/trained_system.joblib
export TENNIS_METRICS_PATH=/path/to/metrics.csv
export TENNIS_PREDICTIONS_PATH=/path/to/predictions.parquet
streamlit run streamlit_app.py
```

> The original Jeff Sackmann ATP/WTA repositories may not always be available. Keep an attributable local or mirrored copy of the source data and review the CC BY-NC-SA terms before publishing or commercializing the app.
