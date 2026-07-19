Tennis live board empty-odds hotfix

Fixes KeyError: 'fixture_id' when no sportsbook/Polymarket quotes are available or aligned.

From the existing project root:
  unzip -o ~/Downloads/tennis-live-board-empty-odds-hotfix.zip -d .
  source .venv/bin/activate
  python -m pip install -e "[dev]"  # use .[dev], see main instructions

Correct reinstall command:
  python -m pip install -e ".[dev]"

Then restart Streamlit:
  python -m streamlit run streamlit_app.py
