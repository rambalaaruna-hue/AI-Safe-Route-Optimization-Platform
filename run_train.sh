#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  echo "Virtual environment missing. Run ./run_setup.sh first."
  exit 1
fi
source .venv/bin/activate
if [[ ! -f real_accident_data.csv ]]; then
  echo "Dataset not found. Generating real_accident_data.csv ..."
  python generate_dataset.py
fi
echo "Training XGBoost risk model..."
python train_model.py
