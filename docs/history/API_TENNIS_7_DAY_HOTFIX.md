# API-Tennis seven-day range hotfix

This overlay changes only `ApiTennisClient.fetch_fixtures()` and adds one test.
Long date spans are split into inclusive seven-calendar-day requests and combined.
Existing match IDs and downstream deduplication remain unchanged.
