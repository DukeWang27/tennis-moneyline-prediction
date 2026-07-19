from tennis_predictor.storage import IdentityStore


def test_cleaned_source_ids_bypass_fuzzy_review(tmp_path):
    store = IdentityStore(tmp_path / "identity.db")

    first = store.resolve_or_create_source_player(
        source="cleaned_kaggle_unified",
        source_player_id="WTA_TEST:wang-xin",
        raw_name="Wang Xin.",
        tour="WTA",
    )

    similar_name = store.resolve_or_create_source_player(
        source="cleaned_kaggle_unified",
        source_player_id="WTA_TEST:wang-xiyu",
        raw_name="Wang Xiy.",
        tour="WTA",
    )

    repeated = store.resolve_or_create_source_player(
        source="cleaned_kaggle_unified",
        source_player_id="WTA_TEST:wang-xin",
        raw_name="Wang Xin.",
        tour="WTA",
    )

    assert first.resolved
    assert similar_name.resolved
    assert repeated.resolved

    assert (
        first.canonical_player_id
        != similar_name.canonical_player_id
    )

    assert (
        first.canonical_player_id
        == repeated.canonical_player_id
    )
