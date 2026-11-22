import pandas as pd

from tennis_predictor.data.names import full_name_keys, surname_initial_key
from tennis_predictor.data.odds import OddsAligner
from tennis_predictor.data.tennis_data import TennisDataLoader


def test_bookmaker_names_match_full_names():
    pairs = {
        "Popyrin A.": "Alexei Popyrin",
        "O Connell C.": "Christopher O'Connell",
        "Auger-Aliassime F.": "Felix Auger Aliassime",
        "De Minaur A.": "Alex de Minaur",
        "Kwon S.W.": "Soonwoo Kwon",
        "Zhang Zh.": "Zhizhen Zhang",
        "Mpetshi Perricard G.": "Giovanni Mpetshi Perricard",
    }
    for short, full in pairs.items():
        assert surname_initial_key(short) in full_name_keys(full), short


def test_brothers_are_not_merged():
    assert surname_initial_key("Zverev M.") not in full_name_keys("Alexander Zverev")
    assert surname_initial_key("Nadal") is None  # not in "Surname I." form: left to the fuzzy pass


def test_month_first_dates_and_women_files(tmp_path):
    # A re-exported women's file: month-first dates, a "WTA" column, and a filename without "wta" in it.
    pd.DataFrame(
        {
            "WTA": [1, 1, 2],
            "Tournament": ["A", "A", "B"],
            "Date": ["1/7/2021", "1/9/2021", "2/1/2021"],
            "Winner": ["Halep S.", "Halep S.", "Swiatek I."],
            "Loser": ["Kerber A.", "Osaka N.", "Sakkari M."],
            "PSW": [1.5, 1.8, 1.3],
            "PSL": [2.6, 2.0, 3.5],
            "B365W": [1.4, 1.7, 1.25],
            "B365L": [2.8, 2.1, 3.75],
        }
    ).to_csv(tmp_path / "2021f.csv", index=False)
    snapshots, report = TennisDataLoader(tmp_path, bookmakers=["pinnacle"]).load()
    assert report.odds_rows == 3
    assert set(snapshots["tour"]) == {"WTA"}
    assert set(snapshots["bookmaker"]) == {"pinnacle"}
    assert snapshots["commence_time"].dt.strftime("%Y-%m-%d").tolist() == ["2021-01-07", "2021-01-09", "2021-02-01"]


def test_key_alignment_orients_odds_to_player1():
    matches = pd.DataFrame(
        {
            "match_id": ["m1", "m2"],
            "date": pd.to_datetime(["2023-05-01", "2023-05-01"]),
            "tour": ["ATP", "ATP"],
            "tournament": ["Madrid", "Madrid"],
            "player1": ["Mischa Zverev", "Daniil Medvedev"],
            "player2": ["Casper Ruud", "Alexander Zverev"],
        }
    )
    snapshots = pd.DataFrame(
        {
            "snapshot_time": pd.to_datetime(["2023-05-02 11:59"]),
            "commence_time": pd.to_datetime(["2023-05-02 12:00"]),
            "tour": ["ATP"],
            "tournament": ["Mutua Madrid Open"],
            "player1": ["Zverev A."],  # listed first by the bookmaker, but player2 in our data
            "player2": ["Medvedev D."],
            "bookmaker": ["pinnacle"],
            "decimal_odds1": [2.5],
            "decimal_odds2": [1.6],
        }
    )
    output, aligned = OddsAligner().align(matches, snapshots)
    assert aligned["match_id"].tolist() == ["m2"]
    row = output.set_index("match_id").loc["m2"]
    assert row["market_decimal1_close"] == 1.6  # Medvedev's price
    assert row["market_decimal2_close"] == 2.5
    assert pd.isna(output.set_index("match_id").loc["m1", "market_decimal1_close"])
