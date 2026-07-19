import pandas as pd

from tennis_predictor.services.live_data import _prepare_moneyline_markets


def test_keeps_only_direct_match_moneylines_and_deduplicates_event():
    markets = pd.DataFrame(
        [
            {
                "market_id": "winner-full",
                "provider_event_id": "event-1",
                "question": "Alice Smith vs Bob Jones",
                "display_title": "Alice Smith vs Bob Jones",
                "player1": "Alice Smith",
                "player2": "Bob Jones",
                "parse_status": "PARSED",
                "tour": "ATP",
                "tier": "ITF M15",
                "tournament": "M15 Test",
                "commence_time": "2026-07-19T18:00:00Z",
                "liquidity": 50,
            },
            {
                "market_id": "winner-short",
                "provider_event_id": "event-1",
                "question": "Smith vs Jones",
                "display_title": "Alice Smith vs Bob Jones",
                "player1": "Smith",
                "player2": "Jones",
                "parse_status": "NEEDS MAPPING",
                "tour": "ATP",
                "tier": "ITF M15",
                "tournament": "M15 Test",
                "commence_time": "2026-07-19T18:00:00Z",
                "liquidity": 500,
            },
            {
                "market_id": "total-games",
                "provider_event_id": "event-1",
                "question": "Total games over or under 20.5",
                "display_title": "Alice Smith vs Bob Jones",
                "player1": "Over 20.5",
                "player2": "Under 20.5",
                "parse_status": "NEEDS MAPPING",
                "tour": "ATP",
                "tier": "ITF M15",
                "tournament": "M15 Test",
                "commence_time": "2026-07-19T18:00:00Z",
            },
            {
                "market_id": "set-winner",
                "provider_event_id": "event-1",
                "question": "First set winner",
                "display_title": "Alice Smith vs Bob Jones",
                "player1": "Alice Smith",
                "player2": "Bob Jones",
                "parse_status": "PARSED",
                "tour": "ATP",
                "tier": "ITF M15",
                "tournament": "M15 Test",
                "commence_time": "2026-07-19T18:00:00Z",
            },
        ]
    )

    result, diagnostics = _prepare_moneyline_markets(
        markets,
        now=pd.Timestamp("2026-07-19T12:00:00Z"),
    )

    assert result["market_id"].tolist() == ["winner-full"]
    assert diagnostics["genuine_moneylines"] == 1
    assert diagnostics["props_filtered"] == 2
    assert diagnostics["duplicate_moneylines_removed"] == 1


def test_live_moneyline_remains_visible_as_market_only_candidate():
    markets = pd.DataFrame(
        [
            {
                "market_id": "live-winner",
                "provider_event_id": "event-live",
                "question": "Matt Hulme vs Jake Delaney",
                "display_title": "Matt Hulme vs Jake Delaney",
                "player1": "Matt Hulme",
                "player2": "Jake Delaney",
                "parse_status": "PARSED",
                "tour": "ATP",
                "tier": "ITF M25",
                "tournament": "M25 Brisbane",
                "commence_time": "2026-07-19T10:00:00Z",
            }
        ]
    )

    result, diagnostics = _prepare_moneyline_markets(
        markets,
        now=pd.Timestamp("2026-07-19T11:00:00Z"),
    )

    assert len(result) == 1
    assert result.iloc[0]["market_phase"] == "LIVE"
    assert bool(result.iloc[0]["model_supported"])
    assert diagnostics["live_moneylines"] == 1


def test_itf_women_moneyline_is_shown_but_marked_unsupported():
    markets = pd.DataFrame(
        [
            {
                "market_id": "wta-itf-winner",
                "provider_event_id": "event-wta-itf",
                "question": "Elizabeth Mandlik vs Kayla Cross",
                "display_title": "Elizabeth Mandlik vs Kayla Cross",
                "player1": "Elizabeth Mandlik",
                "player2": "Kayla Cross",
                "parse_status": "PARSED",
                "tour": "WTA",
                "tier": "W75",
                "tournament": "ITF W75 Granby",
                "commence_time": "2026-07-19T18:00:00Z",
            }
        ]
    )

    result, _ = _prepare_moneyline_markets(
        markets,
        now=pd.Timestamp("2026-07-19T11:00:00Z"),
    )

    assert len(result) == 1
    assert not bool(result.iloc[0]["model_supported"])
    assert "ITF Women" in result.iloc[0]["unsupported_reason"]
