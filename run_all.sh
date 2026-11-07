#!/usr/bin/env bash
# Reproduces every number and figure in the TuringIRL Phase 1 proposal (~5 min, one CPU core).
set -euo pipefail
cd "$(dirname "$0")"
python3 -m pytest -q tests
python3 scripts/run_baselines.py
python3 scripts/run_qae.py
python3 scripts/run_mps.py
python3 scripts/run_resources.py
python3 scripts/make_figures.py
python3 scripts/summary.py
