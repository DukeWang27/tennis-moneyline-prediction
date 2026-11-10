"""Data loaders and adapters.

The core (Sackmann/Kaggle loading, odds alignment, consolidation, the historical backfill) always imports. The live and
sync modules (``kaggle_sync``, ``api_tennis``, ``surface_audit``, ``itf_women_sync``) are optional at import time: if one
is missing, its names are left out and listed in ``MISSING_OPTIONAL_MODULES`` instead of breaking every import of the
package (which is what happened when a ``data/`` gitignore pattern kept them out of the repository). Anything that needs
them raises a clear error when it is used, not when something unrelated is imported.
"""

from __future__ import annotations

MISSING_OPTIONAL_MODULES: list[str] = []

from .kaggle_adapters import (
    BaseKaggleAdapter,
    DissfyaDailyAdapter,
    HugeTennisAdapter,
    KaggleNormalizeReport,
    SchemaContract,
    adapter_for,
    read_tabular_archive,
)

try:
    from .kaggle_sync import (
        KaggleCliClient,
        KaggleDatasetDefinition,
        KaggleRemoteFile,
        KaggleRemoteManifest,
        KaggleSyncResult,
        KaggleSyncService,
        default_dataset_definitions,
    )
except ModuleNotFoundError as e:  # pragma: no cover - depends on the checkout
    if e.name != f"{__name__}.kaggle_sync":
        raise
    MISSING_OPTIONAL_MODULES.append("kaggle_sync")

from .match_consolidation import (
    CanonicalMatchConsolidator,
    ConsolidationReport,
    derive_competition_segment,
    match_fingerprint,
)
from .historical_backfill import (
    ApiTennisHistoryParser,
    ApiTennisProfileHydrator,
    ApiTennisRawDownloader,
    HistoricalDownloadReport,
    HistoricalParseReport,
    classify_event,
    iter_seven_day_windows,
)

try:
    from .api_tennis import (
        ApiTennisClient,
        ProviderPlayerResolver,
        TournamentRegistry,
        normalize_api_tennis_fixtures,
        normalize_api_tennis_settlements,
    )
except ModuleNotFoundError as e:  # pragma: no cover - depends on the checkout
    if e.name != f"{__name__}.api_tennis":
        raise
    MISSING_OPTIONAL_MODULES.append("api_tennis")

from .names import NameResolver, normalize_name
from .odds import OddsAligner, TheOddsApiClient, load_normalized_snapshots
from .sackmann import SackmannLoader

try:
    from .surface_audit import SurfaceAuditReport, run_surface_audit
except ModuleNotFoundError as e:  # pragma: no cover - depends on the checkout
    if e.name != f"{__name__}.surface_audit":
        raise
    MISSING_OPTIONAL_MODULES.append("surface_audit")

try:
    from .itf_women_sync import ItfWomenSyncReport, normalize_sackmann_itf_women, sync_itf_women_history
except ModuleNotFoundError as e:  # pragma: no cover - depends on the checkout
    if e.name != f"{__name__}.itf_women_sync":
        raise
    MISSING_OPTIONAL_MODULES.append("itf_women_sync")

from .tennis_data import TennisDataLoader, TennisDataLoadReport

__all__ = [name for name in [
    "MISSING_OPTIONAL_MODULES",
    "BaseKaggleAdapter",
    "DissfyaDailyAdapter",
    "HugeTennisAdapter",
    "KaggleNormalizeReport",
    "SchemaContract",
    "adapter_for",
    "read_tabular_archive",
    "KaggleCliClient",
    "KaggleDatasetDefinition",
    "KaggleRemoteFile",
    "KaggleRemoteManifest",
    "KaggleSyncResult",
    "KaggleSyncService",
    "default_dataset_definitions",
    "CanonicalMatchConsolidator",
    "ConsolidationReport",
    "derive_competition_segment",
    "match_fingerprint",
    "SurfaceAuditReport",
    "sync_itf_women_history",
    "normalize_sackmann_itf_women",
    "ItfWomenSyncReport",
    "run_surface_audit",
    "TennisDataLoadReport",
    "TennisDataLoader",
    "iter_seven_day_windows",
    "classify_event",
    "HistoricalParseReport",
    "HistoricalDownloadReport",
    "ApiTennisRawDownloader",
    "ApiTennisProfileHydrator",
    "ApiTennisHistoryParser",
    "ApiTennisClient",
    "ProviderPlayerResolver",
    "TournamentRegistry",
    "normalize_api_tennis_fixtures",
    "normalize_api_tennis_settlements",
    "NameResolver",
    "normalize_name",
    "OddsAligner",
    "TheOddsApiClient",
    "load_normalized_snapshots",
    "SackmannLoader",
] if name in globals()]
