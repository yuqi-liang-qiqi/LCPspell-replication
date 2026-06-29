"""
Plot supplementary data-generating-check sensitivity profiles using uncorrected R²_disc.

Uncorrected R²_disc = SS_B/SS_T, recovered from stored chance-corrected means
via the inverse affine map used in the main simulation analysis.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_style("white")
plt.rcParams["figure.dpi"] = 500
plt.rcParams["font.size"] = 10


SCRIPT_DIR = Path(__file__).parent
_BUNDLE_DIR = SCRIPT_DIR.parent
if str(_BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(_BUNDLE_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _bundle_paths import BUNDLE_DIR, FIGURES_DIR, RESULTS_SUPPLEMENTARY_DG_CHECKS, SUPPLEMENTARY_DG_CHECKS_DIR
from _sample_size import resolve_n_total
from supplementary_dg_checks_schema import validate_supplementary_dg_checks_results
from plot_helpers import COLORS, FAMILY_ORDER, LEGEND_LABELS, get_family, short_method_label
from simulation.evaluation import (
    chance_corrected_affine_scale,
    disc_from_chance_corrected,
    expected_disc_baseline,
)


OUT_DIR = FIGURES_DIR
OUT_DIR.mkdir(exist_ok=True)
RESULTS_JSON = RESULTS_SUPPLEMENTARY_DG_CHECKS
DEFAULT_CONFIG_JSON = SUPPLEMENTARY_DG_CHECKS_DIR / "config.json"

PLOT_STRAND_LABELS = {
    "event_order": "Event Order",
    "event_timing": "Event Timing",
    "event_inter_duration": "Inter-event Duration",
    "small_perturbation_token": "Small Perturbation (Token)",
    "small_perturbation_boundary": "Small Perturbation (Boundary)",
}


def corrected_to_disc(mean_corr: float, sd_corr: float, n: int) -> tuple[float, float]:
    scale = chance_corrected_affine_scale(n)
    return (
        disc_from_chance_corrected(mean_corr, n),
        sd_corr * scale,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Supplementary data-generating-check figure with uncorrected R²_disc."
    )
    parser.add_argument(
        "--n-per-group",
        type=int,
        default=None,
        help="Sequences per group for the R²_disc affine map (default: read config.json).",
    )
    parser.add_argument(
        "--config-json",
        type=Path,
        default=DEFAULT_CONFIG_JSON,
        help="Supplementary-check config with n_sequences_per_group.",
    )
    args = parser.parse_args()

    if not RESULTS_JSON.exists():
        raise FileNotFoundError(
            f"Missing {RESULTS_JSON}. Run 03_run_supplementary_data_generating_checks.py first."
        )

    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)
    validate_supplementary_dg_checks_results(raw)

    n = resolve_n_total(n_per_group=args.n_per_group, config_path=args.config_json)
    disc_baseline = expected_disc_baseline(n)

    rows = []
    for strand_key, strand_res in raw.items():
        for method, stats in strand_res.items():
            mean_disc, sd_disc = corrected_to_disc(
                mean_corr=stats["mean"],
                sd_corr=stats["std"],
                n=n,
            )
            rows.append(
                {
                    "strand_key": strand_key,
                    "strand": PLOT_STRAND_LABELS.get(strand_key, strand_key),
                    "method": method,
                    "mean_r2_disc": mean_disc,
                    "sd_r2_disc": sd_disc,
                    "family": get_family(method),
                }
            )
    df = pd.DataFrame(rows)

    panel_order = list(PLOT_STRAND_LABELS.keys())
    if set(panel_order) != set(df["strand_key"].unique()):
        raise ValueError(
            f"Expected supplementary strands {panel_order}, "
            f"got {sorted(df['strand_key'].unique())}"
        )
    fig, axes = plt.subplots(2, 3, figsize=(28, 12.5))
    axes = axes.flatten()

    for i, sk in enumerate(panel_order):
        ax = axes[i]
        sd = df[df["strand_key"] == sk].copy()
        sd["family_order"] = sd["family"].map({f: j for j, f in enumerate(FAMILY_ORDER)})
        sd = sd.sort_values(["family_order", "mean_r2_disc"], ascending=[True, False])
        x = np.arange(len(sd))
        for j, row in enumerate(sd.itertuples(index=False)):
            ax.bar(
                x[j],
                row.mean_r2_disc,
                yerr=row.sd_r2_disc,
                color=COLORS.get(row.family, "#9CA3AF"),
                alpha=0.75,
                capsize=2,
                edgecolor="none",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [short_method_label(m) for m in sd["method"]],
            rotation=90,
            fontsize=8,
        )
        ax.set_title(f"({chr(65+i)}) {PLOT_STRAND_LABELS[sk]}", loc="left", fontweight="bold")
        ax.set_ylabel(r"Mean $R^2_{\mathrm{disc}}$")
        ax.axhline(y=disc_baseline, color="#9CA3AF", linewidth=0.8, linestyle="--", alpha=0.8)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        y_min = float((sd["mean_r2_disc"] - sd["sd_r2_disc"]).min())
        y_max = float((sd["mean_r2_disc"] + sd["sd_r2_disc"]).max())
        y_pad = max(0.004, (y_max - y_min) * 0.12)
        y_lo = min(y_min - y_pad, disc_baseline)
        y_hi = max(y_max + y_pad, disc_baseline)
        ax.set_ylim([y_lo, y_hi])

    for k in range(len(panel_order), len(axes)):
        axes[k].set_visible(False)

    legend_elements = [
        mpatches.Patch(
            facecolor=COLORS[f],
            alpha=0.75,
            label=LEGEND_LABELS.get(f, f),
        )
        for f in FAMILY_ORDER
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 1.01),
        frameon=True,
        framealpha=0.9,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_pdf = OUT_DIR / "fig_supplementary_data_generating_checks_raw.pdf"
    out_png = OUT_DIR / "fig_supplementary_data_generating_checks_raw.png"
    plt.savefig(out_pdf, dpi=500, bbox_inches="tight", pad_inches=0.12)
    plt.savefig(out_png, dpi=500, bbox_inches="tight", pad_inches=0.12)
    plt.close()
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
