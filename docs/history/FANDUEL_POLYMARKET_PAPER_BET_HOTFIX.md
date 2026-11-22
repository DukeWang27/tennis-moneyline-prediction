# FanDuel + Polymarket Live-Board Hotfix

This overlay:

- Requests only FanDuel from The Odds API.
- Discovers Polymarket tennis events using the current comma-separated sports `tags` field.
- Keeps stable empty quote columns and adds feed/matching diagnostics.
- Shows a manual paper-bet form for every resolved fixture, even without an automated quote.
- Accepts FanDuel American or decimal odds.
- Accepts Polymarket prices as `0.58` or `58`, plus American/decimal formats.
- Converts saved prices to decimal odds before writing to SQLite.
- Shows conservative research labels: DATA WARNING, NO PRICE, PASS, WATCH, or PAPER-BET CANDIDATE.
- Blocks a positive label when the model state is stale, player history is extremely sparse, or LightGBM and Elo strongly disagree.
- Clarifies that the board is upcoming-only; past completed matches remain in Match Explorer.

The button records a simulated wager only. It never places a real FanDuel or Polymarket order.
