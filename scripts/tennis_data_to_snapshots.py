"""Turn a folder of Tennis-Data CSV files into the odds-snapshot CSV the backtest reads.

Tennis-Data (tennis-data.co.uk) publishes yearly ATP and WTA results with closing odds from several books, including
Pinnacle, whose closing price is the hardest benchmark in tennis. Download the yearly files (or any re-exported copy),
put them in one folder, then:

    python scripts/tennis_data_to_snapshots.py path/to/tennis_data_csvs data/odds/pinnacle_closing.csv

and set ``odds.normalized_snapshots_path: data/odds/pinnacle_closing.csv`` in config.yaml. Only CSV files are read;
save .xlsx files as CSV first.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tennis_predictor.data.tennis_data import TennisDataLoader


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("source_dir")
    parser.add_argument("output_csv")
    parser.add_argument("--bookmaker", action="append", default=None,
                        help="Book(s) to keep: pinnacle (default), bet365, market_max, market_average, ...")
    args = parser.parse_args(argv)
    snapshots, report = TennisDataLoader(args.source_dir, bookmakers=args.bookmaker or ["pinnacle"]).load()
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    snapshots.drop(columns=["source_file"], errors="ignore").to_csv(args.output_csv, index=False)
    print(f"{report.files} files, {report.source_rows:,} matches, {report.odds_rows:,} prices -> {args.output_csv}")
    if not snapshots.empty:
        print(snapshots.groupby("tour").size().to_string())
        print(f"{snapshots['commence_time'].min():%Y-%m-%d} to {snapshots['commence_time'].max():%Y-%m-%d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
