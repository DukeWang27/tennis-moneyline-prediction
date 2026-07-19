# Rebuild FK + selected-day scanner hotfix

This overlay fixes two issues:

1. During `tennis-rebuild-live`, `match_history` rows are inserted before their dependent `prediction_ledger` rows, satisfying the SQLite foreign key.
2. The Polymarket scanner includes a Match date selector and displays only markets whose local start date matches that selected date in the configured timezone.

No LightGBM, Elo, feature, calibration, or walk-forward mathematics are changed.
