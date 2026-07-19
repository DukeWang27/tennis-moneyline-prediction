import pandas as pd

from tennis_predictor.data.tennis_data import TennisDataLoader


def test_tennis_data_csv_normalizes_historical_odds(tmp_path):
    source = pd.DataFrame([
        {
            "Date": "16/07/2026",
            "Tournament": "Lincoln Challenger",
            "Winner": "Colton Smith",
            "Loser": "Hayato Matsuoka",
            "B365W": 1.30,
            "B365L": 3.40,
        }
    ])
    source.to_csv(tmp_path / "atp_2026.csv", index=False)
    snapshots, report = TennisDataLoader(tmp_path).load()
    assert report.files == 1
    assert report.odds_rows == 1
    assert snapshots.iloc[0]["player1"] == "Colton Smith"
    assert snapshots.iloc[0]["decimal_odds1"] == 1.30
    assert snapshots.iloc[0]["is_closing"]
