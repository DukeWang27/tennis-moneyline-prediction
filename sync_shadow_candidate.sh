#!/bin/bash
set -e

cd "$HOME/Downloads/tennis-moneyline-prediction"
source .venv/bin/activate

tennis-sync \
  --config config.shadow-candidate.yaml \
  --model outputs/shadow/candidate_live.joblib
