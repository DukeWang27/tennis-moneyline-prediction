from tennis_predictor.storage import IdentityStore


def test_source_specific_ids_map_to_one_canonical_player(tmp_path):
    store = IdentityStore(tmp_path / "identity.db")
    store.upsert_player(
        101,
        canonical_name="Ya-Yi Yang",
        tour="WTA",
        dob="2004-01-01",
        nationality="TPE",
    )
    api = store.resolve_source_id(source="API-Tennis", source_player_id="101")
    kaggle = store.resolve_or_create_source_player(
        source="Kaggle-WTA-Daily",
        source_player_id="yang_ya_yi",
        raw_name="Ya Yi Yang",
        tour="WTA",
    )
    assert api.resolved and kaggle.resolved
    assert api.canonical_player_id == kaggle.canonical_player_id
    assert len(store.list_players()) == 1
    assert store.resolve_source_id(
        source="Kaggle-WTA-Daily", source_player_id="yang_ya_yi"
    ).canonical_player_id == api.canonical_player_id
