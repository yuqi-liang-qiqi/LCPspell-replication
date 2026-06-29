"""
@Author  : Yuqi Liang 梁彧祺
@File    : quick_sequencing_check.py
@Time    : 16/02/2026 12:30
@Desc    :
Development utility only. Not part of the replication workflow.

Optional one-panel bar chart: quick check of sequencing (complete_reversal).
For the full multi-panel paper figure, use ``pipeline/02_plot_sensitivity_profiles.py``.

Display rule:
- Internal key: OMspellRS_expcost_*
- Figure label uses OMspellRS (OMsRS)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_BUNDLE = Path(__file__).resolve().parents[2]
if str(_BUNDLE) not in sys.path:
    sys.path.insert(0, str(_BUNDLE))

from simulation.method_labels import method_display_name

DEFAULT_RESULTS_JSON = _BUNDLE / "results" / "main_raw" / "results.json"


def load_tidy_rows(results_json: Path):
    data = json.loads(results_json.read_text(encoding="utf-8"))
    rows = []
    for strand, value in data.items():
        if strand == "sequencing":
            for pattern, methods in value.items():
                for m, stats in methods.items():
                    rows.append(
                        {
                            "strand": f"sequencing:{pattern}",
                            "method_raw": m,
                            "method_display": method_display_name(m),
                            "mean_r2": stats["mean"],
                        }
                    )
        else:
            for m, stats in value.items():
                rows.append(
                    {
                        "strand": strand,
                        "method_raw": m,
                        "method_display": method_display_name(m),
                        "mean_r2": stats["mean"],
                    }
                )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dev-only quick sequencing panel preview."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_RESULTS_JSON,
        help="Path to results.json (default: results/main_raw/results.json).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_path = args.input.resolve()
    if not results_path.is_file():
        raise FileNotFoundError(
            f"Missing {results_path}. Run simulation/run_main.py --norm none first, "
            "or pass --input explicitly."
        )

    out = _BUNDLE / "figures_preview" / "quick_sensitivity_sequencing_check.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = load_tidy_rows(results_path)
    sub = df[df["strand"] == "sequencing:complete_reversal"].copy()
    sub = sub.sort_values("mean_r2", ascending=False).head(16)

    plt.figure(figsize=(12, 4.8))
    plt.bar(range(len(sub)), sub["mean_r2"])
    plt.xticks(range(len(sub)), sub["method_display"], rotation=90, fontsize=8)
    plt.ylabel("Mean pseudo-R2")
    plt.title("Quick check: sequencing complete_reversal")
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
