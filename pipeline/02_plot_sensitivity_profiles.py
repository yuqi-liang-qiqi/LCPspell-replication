"""
Sensitivity profiles figure (main.tex): Panels A--J.

Study 1 (A--F): chance-corrected pseudo-R² — group-level strands.
Study 2 (G--J): normalized aggregate paired contrast — early-vs-late pairs.

Input: aggregated CSV from 01_aggregate_json_to_csv.py (or results.json via results_loader).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SCRIPT_DIR = Path(__file__).parent
_BUNDLE_DIR = SCRIPT_DIR.parent
if str(_BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(_BUNDLE_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _bundle_paths import DATA_MAIN_RAW, DATA_DIR, FIGURES_DIR
from plot_helpers import COLORS, FAMILY_ORDER, get_family, short_method_label
from results_loader import load_results_json, validate_figure_frame
from strand_layout import (
    EXPECTED_PANEL_LETTERS,
    METRIC_CHANCE_CORRECTED_PSEUDO_R2,
    METRIC_NORMALIZED_PAIRED_CONTRAST,
    METRIC_SHORT_YLABEL,
    PANEL_METRICS,
    PANEL_ORDER,
)

sns.set_style("white")
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 8
plt.rcParams["figure.dpi"] = 500

FIGURES_DIR.mkdir(exist_ok=True)

EXPECTED_N_PANELS = len(EXPECTED_PANEL_LETTERS)


def _ylabel_for_strand(strand_name: str) -> str:
    metric = PANEL_METRICS.get(strand_name, METRIC_CHANCE_CORRECTED_PSEUDO_R2)
    return METRIC_SHORT_YLABEL[metric]


def _data_driven_ylim(strand_data: pd.DataFrame, score_col: str, sd_col: str) -> tuple[float, float]:
    lower = float((strand_data[score_col] - strand_data[sd_col]).min())
    upper = float((strand_data[score_col] + strand_data[sd_col]).max())
    span = upper - lower
    pad = max(0.02, 0.08 * (span if span > 0 else 1.0))
    return lower - pad, upper + pad


def create_figure(df: pd.DataFrame, output_suffix: str = "") -> None:
    validate_figure_frame(df)

    score_col = "mean_score" if "mean_score" in df.columns else "mean_r2"
    sd_col = "sd_score" if "sd_score" in df.columns else "sd_r2"

    panel_order_used = [s for s in PANEL_ORDER if s in df["strand"].unique()]
    n_panels = len(panel_order_used)
    if n_panels != EXPECTED_N_PANELS:
        raise ValueError(
            f"Expected {EXPECTED_N_PANELS} panels in data, got {n_panels}."
        )

    fig, axes_grid = plt.subplots(2, 5, figsize=(34, 12))
    axes_flat = list(axes_grid.flat)

    for idx, strand_name in enumerate(panel_order_used):
        ax = axes_flat[idx]
        strand_data = df[df["strand"] == strand_name].copy()
        strand_data["family"] = strand_data["method"].apply(get_family)
        strand_data["family_order"] = strand_data["family"].map(
            {fam: i for i, fam in enumerate(FAMILY_ORDER)}
        )
        strand_data = strand_data.sort_values(["family_order", score_col], ascending=[True, False])

        x_pos = np.arange(len(strand_data))
        for bar_idx, row in enumerate(strand_data.itertuples(index=False)):
            color = COLORS.get(row.family, "#808080")
            ax.bar(
                x_pos[bar_idx],
                getattr(row, score_col),
                yerr=getattr(row, sd_col),
                color=color,
                alpha=0.7,
                edgecolor="none",
                capsize=3,
                error_kw={"elinewidth": 1, "capthick": 1},
            )

        ax.set_xlabel("Method", fontsize=10)
        ax.set_ylabel(_ylabel_for_strand(strand_name), fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [short_method_label(m) for m in strand_data["method"]],
            rotation=90,
            ha="center",
            va="top",
            fontsize=8,
        )

        metric = PANEL_METRICS.get(strand_name, METRIC_CHANCE_CORRECTED_PSEUDO_R2)
        y_lo, y_hi = _data_driven_ylim(strand_data, score_col, sd_col)
        ax.set_ylim([y_lo, y_hi])
        if metric == METRIC_NORMALIZED_PAIRED_CONTRAST:
            ax.axhline(y=1.0, color="lightgray", linestyle="--", linewidth=0.6)
            ax.axhline(y=-1.0, color="lightgray", linestyle="--", linewidth=0.6)

        ax.grid(False)
        ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        panel_label = f"({strand_data['panel'].iloc[0]})"
        ax.text(
            0.02,
            0.98,
            panel_label,
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    legend_labels = {"Others": "Hamming"}
    legend_elements = [
        mpatches.Patch(facecolor=COLORS[family], alpha=0.7, label=legend_labels.get(family, family))
        for family in FAMILY_ORDER
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 1.01),
        frameon=True,
        framealpha=0.9,
        edgecolor="gray",
        fancybox=True,
    )

    plt.tight_layout()

    output_pdf = FIGURES_DIR / f"fig_sensitivity_profiles{output_suffix}.pdf"
    output_png = FIGURES_DIR / f"fig_sensitivity_profiles{output_suffix}.png"
    plt.savefig(output_pdf, dpi=500, bbox_inches="tight", pad_inches=0.12)
    plt.savefig(output_png, dpi=500, bbox_inches="tight", pad_inches=0.12)
    print(f"Figure saved to {output_pdf} and {output_png}")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot sensitivity profiles (Study 1 + Study 2, panels A–J)."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=str(DATA_MAIN_RAW),
        help="Aggregated results CSV.",
    )
    parser.add_argument(
        "--input-json",
        type=str,
        default="",
        help="Optional results.json (overrides --input-csv).",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help="Suffix for output filename, e.g. '_builtin'.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.input_json:
        df = load_results_json(args.input_json)
    else:
        input_csv = Path(args.input_csv)
        if not input_csv.exists():
            candidates = sorted(DATA_DIR.glob("*.csv"))
            raise FileNotFoundError(
                f"Input CSV not found: {input_csv}\n"
                f"Available: {', '.join(p.name for p in candidates) or '(none)'}"
            )
        df = pd.read_csv(input_csv)
        validate_figure_frame(df)

    print(f"Plotting {len(df)} rows across {df['panel'].nunique()} panels...")
    create_figure(df, output_suffix=args.output_suffix)
    print("Done.")
