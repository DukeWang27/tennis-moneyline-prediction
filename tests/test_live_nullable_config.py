from tennis_predictor.services.live_data import (
    _safe_live_config_int,
)


def test_null_numeric_settings_use_defaults():
    assert _safe_live_config_int(
        None,
        default=60,
    ) == 60

    assert _safe_live_config_int(
        "",
        default=20,
    ) == 20

    assert _safe_live_config_int(
        "100",
        default=20,
    ) == 100


def test_numeric_settings_respect_minimum():
    assert _safe_live_config_int(
        0,
        default=60,
    ) == 1

    assert _safe_live_config_int(
        -5,
        default=60,
    ) == 1
