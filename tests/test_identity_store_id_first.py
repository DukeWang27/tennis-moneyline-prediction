from tennis_predictor.storage.identity_store import IdentityStore


def test_api_player_id_unifies_spelling_variants(tmp_path):
    store = IdentityStore(tmp_path / "identity.db")
    store.upsert_player(
        123,
        canonical_name="Ya-Yi Yang",
        tour="ITF_W",
        aliases=["Ya Yi Yang", "Yang Ya Yi"],
    )
    store.link_source_id(
        source="api_tennis",
        source_player_id="123",
        provider_player_id=123,
    )

    exact = store.resolve_name(
        "Ya Yi Yang", source="polymarket", tour="ITF_W"
    )
    reverse = store.resolve_name(
        "Yang Ya Yi", source="polymarket", tour="ITF_W"
    )
    source_id = store.resolve_source_id(source="api_tennis", source_player_id="123")

    assert exact.provider_player_id == 123
    assert reverse.provider_player_id == 123
    assert source_id.provider_player_id == 123
    assert store.model_player_id(123) == "api_tennis:123"


def test_unresolved_name_enters_review_queue(tmp_path):
    store = IdentityStore(tmp_path / "identity.db")
    store.upsert_player(1, canonical_name="Alice Example", tour="WTA")
    resolution = store.resolve_name(
        "Completely Different",
        source="polymarket",
        tour="WTA",
        fuzzy_threshold=99,
    )
    assert not resolution.resolved
    assert resolution.status == "REVIEW_REQUIRED"
    assert len(store.pending_reviews()) == 1


def test_main_tour_lookup_can_resolve_itf_identity(tmp_path):
    store = IdentityStore(tmp_path / "identity.db")
    store.upsert_player(321, canonical_name="Lower Tier Player", tour="ITF_W")
    resolution = store.resolve_name(
        "Lower Tier Player", source="polymarket", tour="WTA"
    )
    assert resolution.provider_player_id == 321
