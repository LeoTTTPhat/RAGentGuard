#!/usr/bin/env bash
# reproduce.sh -- check this release without GPUs or downloads:
#   1. run the unit tests;
#   2. regenerate every table and number of the manuscript from the stored outputs in output/.
# Re-running the experiments themselves needs the models and benchmarks listed in README.md.
set -euo pipefail
PYTHON=${PYTHON:-python3}
cd "$(dirname "$0")"
MAKE_RESULTS=paper/make_results.py
[ -f "$MAKE_RESULTS" ] || MAKE_RESULTS=../paper/make_results.py   # manuscript tree layout
echo "[1/2] Unit tests"
PYTHONPATH=src "$PYTHON" -m pytest -q -p no:cacheprovider tests
echo "[2/2] Tables and numbers of the manuscript from output/"
"$PYTHON" "$MAKE_RESULTS"
echo "Done: LaTeX tables and macros are in $(dirname "$MAKE_RESULTS")/generated/."
