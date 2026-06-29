"""
Plot supplementary data-generating-check sensitivity profiles.
"""

from __future__ import annotations

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
from _bundle_paths import FIGURES_DIR, RESULTS_SUPPLEMENTARY_DG_CHECKS
from supplementary_dg_checks_schema import validate_supplementary_dg_checks_results
from plot_helpers import COLORS, FAMILY_ORDER, LEGEND_LABELS, get_family, short_method_label

OUT_DIR = FIGURES_DIR
OUT_DIR.mkdir(exist_ok=True)
RESULTS_JSON = RESULTS_SUPPLEMENTARY_DG_CHECKS

PLOT_STRAND_LABELS = {
    "event_order": "Event Order",
    "event_timing": "Event Timing",
    "event_inter_duration": "Inter-event Duration",
    "small_perturbation_token": "Small Perturbation (Token)",
    "small_perturbation_boundary": "Small Perturbation (Boundary)",
}


def main() -> None:
    if not RESULTS_JSON.exists():
        raise FileNotFoundError(
            f"Missing {RESULTS_JSON}. Run 03_run_supplementary_data_generating_checks.py first."
        )

    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)
    validate_supplementary_dg_checks_results(raw)

    rows = []
    for strand_key, strand_res in raw.items():
        for method, stats in strand_res.items():
            rows.append(
                {
                    "strand_key": strand_key,
                    "strand": PLOT_STRAND_LABELS.get(strand_key, strand_key),
                    "method": method,
                    "mean_r2": stats["mean"],
                    "sd_r2": stats["std"],
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
        sd = sd.sort_values(["family_order", "mean_r2"], ascending=[True, False])
        x = np.arange(len(sd))
        for j, row in enumerate(sd.itertuples(index=False)):
            ax.bar(
                x[j],
                row.mean_r2,
                yerr=row.sd_r2,
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
        ax.set_ylabel("Mean Pseudo-R² (chance-corrected)")
        ax.axhline(y=0, color="gray", linewidth=0.8)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

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
    out_pdf = OUT_DIR / "fig_supplementary_data_generating_checks.pdf"
    out_png = OUT_DIR / "fig_supplementary_data_generating_checks.png"
    plt.savefig(out_pdf, dpi=500, bbox_inches="tight", pad_inches=0.12)
    plt.savefig(out_png, dpi=500, bbox_inches="tight", pad_inches=0.12)
    plt.close()
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
