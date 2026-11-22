# Timezone Sample-Quality Hotfix

Fixes `Cannot subtract tz-naive and tz-aware datetime-like objects`.

Both stored player-state dates and Polymarket match timestamps are converted
to timezone-naive UTC calendar days before staleness is calculated.

This does not change LightGBM, Elo, probabilities, or Polymarket pricing.
