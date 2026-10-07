#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  echo "Virtual environment missing. Run ./run_setup.sh first."
  exit 1
fi
if [[ ! -f models/accident_model.pkl ]]; then
  echo "Model missing. Run ./run_train.sh first."
  exit 1
fi
source .venv/bin/activate
echo "Starting Streamlit on http://localhost:8501"
streamlit run app.py --server.headless true
