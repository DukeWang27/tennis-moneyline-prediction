# Calendar-Day Timezone Hotfix

The prior timezone patch converted a New York timestamp such as
`2026-07-16 20:30-04:00` to `2026-07-17 00:30 UTC`, incorrectly adding one
day to the sample-staleness calculation.

This version removes timezone metadata while preserving the displayed local
calendar date before subtracting days.

It changes only sample-quality staleness calculations. It does not change:
- LightGBM
- Elo
- model probabilities
- Polymarket prices
- wallet or paper bets
