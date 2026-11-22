"""Skip test files whose modules are not in this checkout yet.

``api_tennis``, ``itf_women_sync``, ``kaggle_sync`` and ``surface_audit`` were kept out of the repository by an old
``data/`` gitignore pattern and still live only in the original working copy. Tests that need them are skipped with
the reason printed (``pytest -rs``), so the rest of the suite still runs. Once the modules are committed, every test
runs again with no change here.
"""

from __future__ import annotations

import importlib.util

_NEEDS = {
    "api_tennis": "tennis_predictor.data.api_tennis",
    "itf_women_sync": "tennis_predictor.data.itf_women_sync",
    "kaggle_sync": "tennis_predictor.data.kaggle_sync",
    "surface_audit": "tennis_predictor.data.surface_audit",
}
_TEST_FILES = {
    "test_api_tennis_chunking.py": ["api_tennis"],
    "test_api_tennis_normalization.py": ["api_tennis"],
    "test_itf_women_sync.py": ["itf_women_sync"],
    "test_kaggle_sync_versioning.py": ["kaggle_sync"],
    "test_surface_registry_audit.py": ["api_tennis", "surface_audit"],
    # services.live_data imports api_tennis
    "test_live_identity_without_provider_id.py": ["api_tennis"],
    "test_live_nullable_config.py": ["api_tennis"],
    "test_live_parquet_datetime_types.py": ["api_tennis"],
    "test_live_price_eligibility_fix.py": ["api_tennis"],
    "test_moneyline_market_filter.py": ["api_tennis"],
    "test_polymarket_selected_day.py": ["api_tennis"],
    "test_rebuild_foreign_key_order.py": ["api_tennis"],
}


def _missing(name: str) -> bool:
    try:
        return importlib.util.find_spec(_NEEDS[name]) is None
    except ModuleNotFoundError:
        return True


MISSING = sorted(name for name in _NEEDS if _missing(name))
collect_ignore = [f for f, needs in _TEST_FILES.items() if any(n in MISSING for n in needs)]


def pytest_report_header(config):
    if MISSING:
        return [f"modules not in this checkout (their {len(collect_ignore)} test files are skipped): {', '.join(MISSING)}"]
    return []
