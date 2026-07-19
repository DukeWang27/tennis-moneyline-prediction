from pathlib import Path

import pandas as pd
import pytest

from tennis_predictor.storage import (
    get_wallet_summary,
    initialize_paper_trading_db,
    list_bets,
    place_paper_bet,
    settle_pending_bets_from_results,
)


def test_wallet_place_and_settle_is_atomic(tmp_path: Path):
    db = tmp_path / "paper_trading.db"
    initialize_paper_trading_db(db)
    assert get_wallet_summary(db).balance == 100.0

    place_paper_bet(
        db,
        fixture_id="api_tennis:1",
        matchup="A vs B",
        bet_on="A",
        odds=2.0,
        sportsbook="Book",
        stake=5.0,
        p_model=0.60,
    )
    assert get_wallet_summary(db).balance == 95.0
    with pytest.raises(ValueError, match="already exists"):
        place_paper_bet(
            db,
            fixture_id="api_tennis:1",
            matchup="A vs B",
            bet_on="A",
            odds=2.0,
            sportsbook="Book",
            stake=5.0,
            p_model=0.60,
        )

    result = pd.DataFrame(
        [{"fixture_id": "api_tennis:1", "player1": "A", "player2": "B", "player1_won": 1}]
    )
    counts = settle_pending_bets_from_results(db, result)
    assert counts == {"won": 1, "lost": 0, "void": 0}
    summary = get_wallet_summary(db)
    assert summary.balance == 105.0
    assert summary.wins == 1
    assert summary.roi == pytest.approx(1.0)
    assert list_bets(db).iloc[0]["status"] == "WON"

    # A repeated result sync cannot credit the wallet twice.
    settle_pending_bets_from_results(db, result)
    assert get_wallet_summary(db).balance == 105.0
