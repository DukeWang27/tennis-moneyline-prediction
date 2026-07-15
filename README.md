# Tennis Moneyline Prediction

A modular Python pipeline for ATP/WTA match ingestion, leakage-safe feature generation, calibrated win probabilities, market comparison, and walk-forward backtesting.

## What is implemented

- ATP and WTA yearly CSV ingestion from Jeff Sackmann's GitHub repositories, including optional ATP qualifying/Challenger and WTA qualifying/ITF files.
- Stable player IDs as the primary identity key, with conservative fuzzy matching for external odds feeds.
- Explicit retirement, walkover, and incomplete-match fields. Walkovers receive zero Elo/model weight by default; retirements receive reduced weight rather than being silently treated like ordinary completed losses.
- Toggleable match flags: round-robin/dead-rubber context, already-qualified context, fatigue, and low-tier event immediately before a Grand Slam.
- Strictly pre-match, streaming features: overall and surface-blended Elo, age, experience, retirement/walkover history, recent form, head-to-head, and rest.
- An empirical spline age curve fitted inside each historical training fold, then expressed as an Elo-equivalent adjustment.
- Maximum-likelihood Elo logistic scale fitting.
- LightGBM probability model with a chronological calibration holdout and isotonic or Platt calibration.
- Two-way de-vigging, model/market edge, line movement, flat-bet ROI, fractional-Kelly ROI, CLV, and bootstrap confidence intervals.
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

## Download and preprocess

```bash
tennis-download --config config.yaml
```

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

## Testing

```bash
pytest
```

The included tests cover odds math, retirement handling, and the central no-leakage invariant: a match cannot influence its own pre-match features.
