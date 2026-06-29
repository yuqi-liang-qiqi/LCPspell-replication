"""Shared paths for pipeline scripts (relative to lcpspell-paper-replication/)."""

from pathlib import Path

BUNDLE_DIR = Path(__file__).resolve().parent.parent
PIPELINE_DIR = BUNDLE_DIR / "pipeline"
DATA_DIR = PIPELINE_DIR / "data"
FIGURES_DIR = BUNDLE_DIR / "figures"
TABLES_DIR = BUNDLE_DIR / "tables"
RESULTS_DIR = BUNDLE_DIR / "results"

# Main text: raw distances (norm=none)
RESULTS_MAIN_RAW = RESULTS_DIR / "main_raw" / "results.json"
DATA_MAIN_RAW = DATA_DIR / "aggregated_results.csv"

# Appendix robustness 1: built-in / norm=auto
RESULTS_ROBUSTNESS_BUILTIN = RESULTS_DIR / "robustness_builtin" / "results.json"
DATA_ROBUSTNESS_BUILTIN = DATA_DIR / "aggregated_results_builtin.csv"

# Appendix robustness 2: Elzinga–Studer reference-based rescaling
RESULTS_ROBUSTNESS_ELZINGA = RESULTS_DIR / "robustness_elzinga" / "results.json"
DATA_ROBUSTNESS_ELZINGA = DATA_DIR / "aggregated_results_elzinga.csv"

SUPPLEMENTARY_DG_CHECKS_DIR = RESULTS_DIR / "supplementary_data_generating_checks"
RESULTS_SUPPLEMENTARY_DG_CHECKS = SUPPLEMENTARY_DG_CHECKS_DIR / "results.json"
