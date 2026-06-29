"""
Convert simulation/run_main.py JSON output into pipeline CSV for sensitivity figures.

Defaults (separate files per normalization mode):
  results/main_raw/results.json              -> pipeline/data/aggregated_results.csv
  results/robustness_builtin/results.json    -> pipeline/data/aggregated_results_builtin.csv
  results/robustness_elzinga/results.json    -> pipeline/data/aggregated_results_elzinga.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
_BUNDLE_DIR = SCRIPT_DIR.parent
if str(_BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(_BUNDLE_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _bundle_paths import (
    DATA_MAIN_RAW,
    DATA_ROBUSTNESS_BUILTIN,
    DATA_ROBUSTNESS_ELZINGA,
    RESULTS_MAIN_RAW,
    RESULTS_ROBUSTNESS_BUILTIN,
    RESULTS_ROBUSTNESS_ELZINGA,
)
from results_loader import load_results_json

NORM_TO_RESULTS = {
    "none": RESULTS_MAIN_RAW,
    "builtin": RESULTS_ROBUSTNESS_BUILTIN,
    "elzinga": RESULTS_ROBUSTNESS_ELZINGA,
}
NORM_TO_CSV = {
    "none": DATA_MAIN_RAW,
    "builtin": DATA_ROBUSTNESS_BUILTIN,
    "elzinga": DATA_ROBUSTNESS_ELZINGA,
}

CSV_COLUMNS = [
    "panel",
    "study",
    "strand_number",
    "strand",
    "strand_key",
    "sequencing_pattern",
    "metric",
    "method",
    "method_raw",
    "mean_score",
    "sd_score",
    "win_mean",
    "win_sd",
    "mean_r2",
    "sd_r2",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate simulation JSON into pipeline CSV for paper figures."
    )
    p.add_argument(
        "--from-norm",
        choices=["none", "builtin", "elzinga"],
        help="Use results/{main_raw|robustness_*}/results.json and default CSV.",
    )
    p.add_argument("--input", type=Path, help="Path to results.json (overrides --from-norm).")
    p.add_argument("--output", type=Path, help="Output CSV (required with --input).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.from_norm and (args.input or args.output):
        raise SystemExit("Use either --from-norm or explicit --input/--output, not both.")

    if args.from_norm:
        in_path = NORM_TO_RESULTS[args.from_norm]
        out_path = NORM_TO_CSV[args.from_norm]
    elif args.input:
        if not args.output:
            raise SystemExit("When using --input, pass --output explicitly.")
        in_path = args.input.resolve()
        out_path = args.output.resolve()
    else:
        raise SystemExit(
            "Specify --from-norm none|builtin|elzinga, or pass --input and --output."
        )

    if not in_path.is_file():
        raise FileNotFoundError(
            f"Missing {in_path}\n"
            "Run simulation/run_main.py --norm none|builtin|elzinga, or pass --input."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_results_json(in_path)
    for col in CSV_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Expected column {col!r}, got: {list(df.columns)}")
    df = df[CSV_COLUMNS]
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(df)} rows) from {in_path.resolve()}")


if __name__ == "__main__":
    main()
