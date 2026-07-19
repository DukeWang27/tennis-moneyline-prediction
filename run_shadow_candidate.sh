#!/bin/bash
set -e

cd "$HOME/Downloads/tennis-moneyline-prediction"
source .venv/bin/activate

export TENNIS_CONFIG_PATH="config.shadow-candidate.yaml"
export TENNIS_PAPER_DB="data/shadow/candidate_paper_trading.db"
export TENNIS_PROCESSED_PATH="data/shadow/candidate_matches.parquet"

python -m streamlit run \
  shadow_streamlit_app.py \
  --server.port 8502
