import pandas as pd

from tennis_predictor.data import adapter_for


def test_daily_adapter_preserves_closing_odds_and_status(tmp_path):
    pd.DataFrame(
        {
            "Date": ["2026-07-01", "2026-07-02"],
            "Tournament": ["Example Open", "Example Open"],
            "Series": ["ATP250", "ATP250"],
            "Court": ["Outdoor", "Outdoor"],
            "Surface": ["Hard", "Hard"],
            "Round": ["R32", "R16"],
            "Winner": ["A Player", "B Player"],
            "Loser": ["B Player", "C Player"],
            "WRank": [10, 20],
            "LRank": [20, 30],
            "AvgW": [1.50, 1.80],
            "AvgL": [2.60, 2.00],
            "Comment": ["Completed", "Retired"],
            "Score": ["6-4 6-4", "6-3 RET"],
        }
    ).to_csv(tmp_path / "atp.csv", index=False)

    frame, report = adapter_for("kaggle_atp_daily").normalize_archive(tmp_path)
    assert report.output_rows == 2
    assert frame.loc[0, "closing_odds_player1"] == 1.50
    assert frame.loc[0, "closing_odds_player2"] == 2.60
    assert frame.loc[0, "closing_odds_source"] == "kaggle_atp_daily"
    assert bool(frame.loc[1, "retirement"])
    assert not bool(frame.loc[1, "game_form_eligible"])


def test_adapter_parses_compact_dates_and_excludes_doubles(tmp_path):
    pd.DataFrame(
        {
            "tourney_date": [20260703, 20260704],
            "tourney_name": ["M25 Example", "M25 Example"],
            "winner_name": ["Solo One", "A One / A Two"],
            "loser_name": ["Solo Two", "B One / B Two"],
            "tourney_level": ["F", "F"],
            "round": ["Q1", "R32"],
            "event_type": ["singles", "doubles"],
        }
    ).to_csv(tmp_path / "matches.csv", index=False)
    frame, _ = adapter_for("huge_tennis").normalize_archive(tmp_path)
    assert len(frame) == 1
    assert frame.loc[0, "date"].date().isoformat() == "2026-07-03"
    assert frame.loc[0, "tier"] == "ITF"
    assert bool(frame.loc[0, "is_qualifying"])
