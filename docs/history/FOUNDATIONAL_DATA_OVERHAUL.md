# Foundational Data Overhaul

This overlay rebuilds the data layer around immutable API-Tennis player IDs while leaving the existing LightGBM estimator, calibration mathematics, Elo formulas, and feature definitions intact.

## Architecture

```text
API-Tennis seven-day raw archive
        |
        v
ID-first normalized results + identity.db
        |
        +--> alias review queue
        +--> tournament surface/tier registry
        |
        v
Sackmann/current processed matches + optional Tennis-Data odds
        |
        v
canonical merge and duplicate removal
        |
        v
strict chronological pre-match feature replay
        |
        v
existing walk-forward LightGBM training
        |
        v
segmented metrics + promotion gate
```

## New commands

### Archive and normalize API-Tennis history

```bash
tennis-backfill-history \
  --config config.yaml \
  --start 2023-01-01 \
  --end 2026-07-16
```

The downloader is resumable. Existing seven-day JSON files are skipped unless `--force` is supplied. Each raw response is atomically written beneath:

```text
data/raw/api_tennis/YYYY/MM/YYYY-MM-DD_YYYY-MM-DD.json
```

Useful modes:

```bash
# Download without parsing
tennis-backfill-history --config config.yaml --download-only

# Reparse the existing raw archive without new API calls
tennis-backfill-history --config config.yaml --parse-only

# Fetch DOB/nationality profiles gradually, subject to API quota
tennis-backfill-history \
  --config config.yaml \
  --parse-only \
  --hydrate-profiles \
  --max-profiles 250
```

### Build and validate a candidate model

```bash
tennis-rebuild-candidate \
  --config config.yaml \
  --start 2023-01-01 \
  --end 2026-07-16
```

Outputs are written to `outputs/candidate/` and do not replace production by default:

```text
trained_system_candidate.joblib
predictions.parquet
metrics.csv
calibration_curves.csv
promotion_gate.json
assembly_report.json
```

Only after reviewing a PASS gate should promotion be considered:

```bash
tennis-rebuild-candidate \
  --config config.yaml \
  --start 2023-01-01 \
  --end 2026-07-16 \
  --promote-if-passed
```

## ID-first identity schema

`data/identity.db` contains:

- `players`: one row per API-Tennis `provider_player_id`
- `player_aliases`: exact text aliases pointing to a provider ID
- `source_player_ids`: Sackmann/Tennis-Data/API IDs linked to the canonical provider ID
- `alias_review_queue`: unresolved or ambiguous fuzzy matches
- `tournament_registry`: persistent tournament surface and tier metadata

Model-facing IDs use the stable form `api_tennis:<provider_player_id>`.

## Match-status behavior

- Completed: normal chronological update.
- Retirement: retained and down-weighted by the existing retirement policy.
- Walkover/default/void: retained for audit, but receives zero state-update weight so it does not distort Elo, form, rest, H2H, or experience.
- Unknown surface: retained; overall Elo/form/rest/H2H update normally, while surface Elo is skipped.

## Leakage protection

Every row receives a deterministic `chronology_ts`. The rebuild:

1. Sorts all canonical matches by `chronology_ts`.
2. Creates both players' features before the result is applied.
3. Saves the pre-match row.
4. Updates the state from the result.
5. Advances to the next match.

No old results are appended onto a newer production state.

## Segmented validation

The report separates:

- ATP Main Tour
- WTA Main Tour
- ATP Challenger
- ITF Men
- ITF Women

Metrics include match count, accuracy, Brier score, log loss, expected calibration error, calibration bins, and ROI where aligned historical odds exist.

The promotion gate requires improved log loss and non-worse calibration for Challenger, ITF Men, and ITF Women, with a configurable minimum number of matches. Missing or undersized segments produce `INCONCLUSIVE`, never an automatic promotion.

## Optional Tennis-Data odds

Place locally downloaded CSV files in a directory such as:

```text
data/raw/tennis_data/
```

Then add to `config.yaml`:

```yaml
history:
  tennis_data_directory: data/raw/tennis_data
```

The loader recognizes common historical moneyline columns and converts them into the project's existing odds snapshots for chronological alignment.

## Streamlit identity review

The Research navigation now includes `Alias Review Queue`. Ambiguous Polymarket or source names can be attached to a verified API-Tennis player ID without editing model state by hand.

## Safety

- Keep the current production artifact until the candidate gate passes.
- Do not use `--force` during normal resume operations.
- Do not promote on aggregate accuracy alone.
- Keep this system in paper-trading mode until segment-specific calibration and ROI are established on a meaningful sample.
