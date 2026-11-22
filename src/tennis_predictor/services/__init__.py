from .candidate_training import (
    CandidateBuildReport,
    ModelGateReport,
    MultiSourceMatchAssembler,
    build_candidate_model,
    derive_competition_segment,
    evaluate_model_gate,
    evaluate_paired_model_gate,
    latest_kaggle_sources,
    prepare_chronological_matches,
)
_LIVE_DATA_NAMES = {
    "LiveBoard",
    "SyncReport",
    "backfill_completed_matches",
    "build_live_board",
    "rebuild_live_state",
    "settle_recent_pending_bets",
}


def __getattr__(name: str):
    # live_data needs the api_tennis module, which is not in the repository yet. Importing it lazily keeps the
    # backtest and prediction commands working without it; the live commands still fail with a clear error.
    if name in _LIVE_DATA_NAMES:
        from . import live_data

        return getattr(live_data, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "evaluate_model_gate",
    "evaluate_paired_model_gate",
    "latest_kaggle_sources",
    "prepare_chronological_matches",
    "derive_competition_segment",
    "build_candidate_model",
    "MultiSourceMatchAssembler",
    "ModelGateReport",
    "CandidateBuildReport",
    "LiveBoard",
    "SyncReport",
    "backfill_completed_matches",
    "build_live_board",
    "rebuild_live_state",
    "settle_recent_pending_bets",
]
