import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from tennis_predictor.services.live_data import (
    _normalize_parquet_datetime_columns,
)


def test_normalizes_mixed_live_datetime_types():
    frame = pd.DataFrame(
        {
            "date": [
                pd.Timestamp("2026-07-13"),
                b"2026-07-14",
                "2026-07-15",
            ],
            "event_date": [
                b"2026-07-13",
                pd.Timestamp("2026-07-14"),
                "2026-07-15",
            ],
            "commence_time": [
                b"2026-07-13 12:30:00",
                pd.Timestamp(
                    "2026-07-14 13:45:00"
                ),
                b"14:30",
            ],
        }
    )

    result = _normalize_parquet_datetime_columns(
        frame
    )

    for column in (
        "date",
        "event_date",
        "commence_time",
    ):
        assert is_datetime64_any_dtype(
            result[column]
        )

    assert result.loc[
        0,
        "commence_time",
    ] == pd.Timestamp("2026-07-13 12:30:00")

    assert result.loc[
        1,
        "commence_time",
    ] == pd.Timestamp("2026-07-14 13:45:00")

    assert result.loc[
        2,
        "commence_time",
    ] == pd.Timestamp("2026-07-15 14:30:00")
