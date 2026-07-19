from tennis_predictor.data.api_tennis import ApiTennisClient


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_fetch_fixtures_splits_long_ranges_into_seven_day_windows(monkeypatch):
    calls = []

    def fake_get(url, *, params, timeout):
        calls.append(params.copy())
        event_key = f"{params['date_start']}:{params['date_stop']}"
        return _Response({"success": 1, "result": [{"event_key": event_key}]})

    monkeypatch.setattr("tennis_predictor.data.api_tennis.requests.get", fake_get)

    rows = ApiTennisClient(api_key="test").fetch_fixtures(
        "2026-06-08",
        "2026-07-15",
    )

    assert [(c["date_start"], c["date_stop"]) for c in calls] == [
        ("2026-06-08", "2026-06-14"),
        ("2026-06-15", "2026-06-21"),
        ("2026-06-22", "2026-06-28"),
        ("2026-06-29", "2026-07-05"),
        ("2026-07-06", "2026-07-12"),
        ("2026-07-13", "2026-07-15"),
    ]
    assert len(rows) == 6
