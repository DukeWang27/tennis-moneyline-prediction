import json
from pathlib import Path

from tennis_predictor.data.historical_backfill import (
    ApiTennisRawDownloader,
    iter_seven_day_windows,
)


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _Session:
    def __init__(self):
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append(dict(params))
        return _Response({"success": 1, "result": [{"event_key": params["date_start"]}]})


def test_seven_day_windows_are_inclusive():
    windows = list(iter_seven_day_windows("2023-01-01", "2023-01-15"))
    assert [(a.date().isoformat(), b.date().isoformat()) for a, b in windows] == [
        ("2023-01-01", "2023-01-07"),
        ("2023-01-08", "2023-01-14"),
        ("2023-01-15", "2023-01-15"),
    ]


def test_downloader_archives_raw_payload_before_parse(tmp_path, monkeypatch):
    monkeypatch.setenv("API_TENNIS_KEY", "test-key")
    session = _Session()
    downloader = ApiTennisRawDownloader(
        session=session,
        sleep_between_requests=0,
        max_retries=0,
    )
    report = downloader.download(
        start_date="2023-01-01",
        end_date="2023-01-08",
        raw_root=tmp_path,
    )
    assert report.downloaded_windows == 2
    files = sorted(Path(tmp_path).glob("**/*.json"))
    assert len(files) == 2
    payload = json.loads(files[0].read_text())
    assert payload["success"] == 1
    assert "APIkey" not in files[0].read_text()
    assert session.calls[0]["date_start"] == "2023-01-01"
    assert session.calls[0]["date_stop"] == "2023-01-07"
