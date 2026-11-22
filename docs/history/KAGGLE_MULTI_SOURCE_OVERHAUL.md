# Kaggle Multi-Source Data Overhaul

This overlay extends the existing API-Tennis historical architecture with three versioned Kaggle sources while leaving the LightGBM estimator, calibration method, Elo formula, and numeric feature list unchanged.

## Data flow

```text
Kaggle remote manifest
  -> versioned raw archive
  -> semantic schema validation
  -> source-specific normalized parquet
  -> canonical player identity resolution
  -> deterministic match fingerprint and field-level merge
  -> candidate-only SQLite + data/processed/matches_full.parquet
  -> strict chronological pre-match feature replay
  -> segmented walk-forward validation
  -> paired statistical promotion gate
```

Production `data/processed/matches.parquet` and `outputs/backtest/trained_system.joblib` are not overwritten during synchronization or candidate building.

## New command: `tennis-sync-kaggle`

Install and authenticate the official Kaggle CLI first:

```bash
python -m pip install -e ".[dev]"
kaggle auth login
```

Then archive and normalize all three sources:

```bash
tennis-sync-kaggle --config config.yaml
```

Source selection is optional:

```bash
tennis-sync-kaggle --config config.yaml --source atp --source wta
tennis-sync-kaggle --config config.yaml --source huge
```

Safety behavior:

- Remote file manifests are hashed before downloading.
- An unchanged accepted manifest is skipped.
- ATP/WTA daily archives are stored under `data/raw/kaggle/<source>/YYYY-MM-DD/<manifest-hash>/`.
- The first Huge Tennis snapshot is stored under `data/raw/kaggle/huge_tennis/initial_snapshot/`.
- A later Huge Tennis revision receives a new snapshot directory; the initial snapshot is never overwritten.
- Normalized versions are immutable and a `LATEST.json` pointer identifies the accepted version.
- A semantic schema change is rejected unless `--accept-schema-change` is supplied after manual review.
- A daily source whose maximum match date did not advance is archived but not promoted to the latest normalized version.

Useful recovery flags:

```bash
tennis-sync-kaggle --config config.yaml --continue-on-error
tennis-sync-kaggle --config config.yaml --force-download
tennis-sync-kaggle --config config.yaml --force-ingest
tennis-sync-kaggle --config config.yaml --accept-schema-change
```

Do not routinely use the force or schema-acceptance flags.

## Canonical identity database

`data/identity.db` now uses:

```text
players
  canonical_player_id INTEGER PRIMARY KEY AUTOINCREMENT
  canonical_name
  dob
  nationality
  tour

player_source_mappings
  canonical_player_id
  source_name
  source_player_id

player_aliases
alias_review_queue
tournament_registry
source_ingestion_state
```

Resolution order:

1. Exact `(source_name, source_player_id)` mapping.
2. Exact verified alias.
3. Exact normalized player name within a compatible tour.
4. High-threshold fuzzy match with a minimum score margin.
5. Ambiguous candidates enter the Streamlit Alias Review Queue.
6. A genuinely new, low-similarity player receives a new canonical ID.

The migration preserves the old API-Tennis-ID tables as renamed legacy tables and creates canonical mappings instead of deleting historical identity rows.

## Match consolidation

The canonical fingerprint is:

```text
SHA-256(match_date | sorted canonical player IDs | normalized tournament | normalized round)
```

Rows are oriented by canonical player ID, not by winner/loser position. This prevents winner-first source layouts from leaking the result into player position.

Field precedence is applied independently:

- Result, score, status: API-Tennis, then ATP/WTA daily, then Huge Tennis.
- Surface, court, tier: ATP/WTA daily, then Huge Tennis, then API-Tennis.
- Closing odds: ATP/WTA daily or Tennis-Data only.

Every merged row stores `sources_present` and per-field provenance. Raw source records remain linked in `match_source_records`.

## Leakage-safe chronological rebuild

Run after Kaggle sync, API-Tennis backfill, and alias review:

```bash
tennis-rebuild-candidate \
  --config config.yaml \
  --start 2023-01-01
```

The candidate build:

1. Loads latest accepted Kaggle snapshots, normalized API-Tennis history, and the existing historical parquet.
2. Resolves canonical identities and removes source duplicates.
3. Writes the candidate-only match store to `data/candidate_matches.db`.
4. Writes consolidated matches to `data/processed/matches_full.parquet`.
5. Sorts by `chronology_ts` and deterministic tie-breakers.
6. Creates pre-match Elo, surface Elo, form, rest, and H2H features.
7. Applies the result only after the pre-match row is saved.
8. Writes `data/processed/features_full.parquet`.
9. Runs the unchanged LightGBM walk-forward pipeline.

Some historical sources only provide a calendar date. Those rows are marked `chronology_precision=date_round_estimated`; known round order plus a deterministic hash is used for a stable within-day sequence. Exact timestamps are marked `exact_timestamp`.

Unknown surfaces remain in the replay. They update overall Elo, form, rest, H2H, and experience, but do not update surface-specific Elo.

Walkovers, defaults, and void matches remain auditable but receive zero state-update weight. Retirements retain the existing down-weighting policy.

## Closing odds isolation

Kaggle closing prices are retained for historical ROI only. They are mapped to the backtest's price columns and labelled `odds_benchmark_type=historical_closing`.

Closing odds, score, status, and match outcome are not in `FeatureEngineer.numeric_feature_columns()`. The candidate builder asserts this before training.

Because these are closing prices rather than guaranteed executable opening prices, the resulting ROI is a closing-price benchmark, not proof of achievable live returns.

## Segmented validation and promotion

Reports remain separated into:

- ATP Main Tour
- WTA Main Tour
- ATP Challenger
- ITF Men
- ITF Women

Outputs:

```text
outputs/candidate/trained_system_candidate.joblib
outputs/candidate/predictions.parquet
outputs/candidate/metrics.csv
outputs/candidate/calibration_curves.csv
outputs/candidate/promotion_gate.json
outputs/candidate/assembly_report.json
outputs/candidate/source_inventory.json
outputs/candidate/identity_unresolved.parquet
```

Promotion uses paired predictions for the exact same matches. For Challenger, ITF Men, and ITF Women, bootstrap confidence intervals are calculated for:

```text
candidate log loss - production log loss
candidate ECE - production ECE
```

A segment passes only when the upper confidence bound for both differences is below zero. Missing overlap or too few paired matches produces `INCONCLUSIVE`; it never promotes automatically.

Production replacement requires both a `PASS` and an explicit command:

```bash
tennis-rebuild-candidate \
  --config config.yaml \
  --start 2023-01-01 \
  --promote-if-passed
```

The previous production model is backed up before replacement.

## Recommended operating schedule

```text
API-Tennis current results: daily
automated ATP/WTA Kaggle check: daily
Huge Tennis snapshot: once, then occasional manifest checks
identity review: after each new import
candidate reconstruction: after major data changes
candidate retraining: monthly or after a meaningful coverage increase
production promotion: only after paired segment gate PASS
```
