#!/usr/bin/env bash
# Regenerate paper figures from bundled JSON/CSV (does not re-run long simulations).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"

echo "=== Aggregate CSVs (separate per normalization mode) ==="
if [[ -f results/main_raw/results.json ]]; then
  "$PY" pipeline/01_aggregate_json_to_csv.py --from-norm none
else
  echo "Skip none: missing results/main_raw/results.json (run: python3 simulation/run_main.py --norm none)"
fi
if [[ -f results/robustness_builtin/results.json ]]; then
  "$PY" pipeline/01_aggregate_json_to_csv.py --from-norm builtin
else
  echo "Skip builtin: missing results/robustness_builtin/results.json"
fi
if [[ -f results/robustness_elzinga/results.json ]]; then
  "$PY" pipeline/01_aggregate_json_to_csv.py --from-norm elzinga
else
  echo "Skip elzinga: missing results/robustness_elzinga/results.json"
fi

echo "=== Sensitivity figures ==="
if [[ -f pipeline/data/aggregated_results.csv ]]; then
  "$PY" pipeline/02_plot_sensitivity_profiles.py \
    --input-csv pipeline/data/aggregated_results.csv
else
  echo "Skip main figure: missing pipeline/data/aggregated_results.csv"
fi
if [[ -f pipeline/data/aggregated_results_builtin.csv ]]; then
  "$PY" pipeline/02_plot_sensitivity_profiles.py \
    --input-csv pipeline/data/aggregated_results_builtin.csv \
    --output-suffix "_builtin"
fi
if [[ -f pipeline/data/aggregated_results_elzinga.csv ]]; then
  "$PY" pipeline/02_plot_sensitivity_profiles.py \
    --input-csv pipeline/data/aggregated_results_elzinga.csv \
    --output-suffix "_elzinga"
fi
if [[ -f pipeline/data/aggregated_results.csv ]]; then
  "$PY" pipeline/02_plot_sensitivity_raw_pseudor2.py \
    --input-csv pipeline/data/aggregated_results.csv
fi

echo "=== Supplementary data-generating checks (figures) ==="
"$PY" pipeline/04_plot_supplementary_data_generating_checks.py
"$PY" pipeline/04_plot_supplementary_data_generating_checks_raw.py

echo "Done. Figures in figures/"
