"""
Generate appendix figure using uncorrected discrepancy-based pseudo-R² (R²_disc).

Study 1 panels only (A–F). Reads aggregated chance-corrected results and maps
them to R²_disc = SS_B/SS_T via the inverse of equation (R²_cc):

    R²_disc = 1/(n-1) + (1 - 1/(n-1)) * R²_cc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_style("white")
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["figure.dpi"] = 500


SCRIPT_DIR = Path(__file__).parent
_BUNDLE_DIR = SCRIPT_DIR.parent
if str(_BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(_BUNDLE_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _bundle_paths import BUNDLE_DIR, DATA_MAIN_RAW, FIGURES_DIR
from _sample_size import ensure_matching_sample_size_source, resolve_n_total
from plot_helpers import COLORS, FAMILY_ORDER, get_family, short_method_label
from results_loader import validate_figure_frame
from simulation.evaluation import (
    chance_corrected_affine_scale,
    disc_from_chance_corrected,
    expected_disc_baseline,
)
from strand_layout import PANEL_SPECS


FIGURES_DIR.mkdir(exist_ok=True)

DEFAULT_INPUT_CSV = DATA_MAIN_RAW
DEFAULT_CONFIG_JSON = BUNDLE_DIR / "results" / "main_raw" / "config.json"
DEFAULT_OUT_BASE = "fig_sensitivity_raw_pseudor2_core"

STUDY1_PANEL_SPECS = [spec for spec in PANEL_SPECS if spec["study"] == "study1"]
STUDY1_STRANDS = [spec["strand_label"] for spec in STUDY1_PANEL_SPECS]


def to_disc_pseudo_r2(df: pd.DataFrame, n: int) -> pd.DataFrame:
    scale = chance_corrected_affine_scale(n)
    out = df.copy()
    score_col = "mean_score" if "mean_score" in out.columns else "mean_r2"
    sd_col = "sd_score" if "sd_score" in out.columns else "sd_r2"
    out["mean_r2_disc"] = out[score_col].map(lambda v: disc_from_chance_corrected(v, n))
    out["sd_r2_disc"] = out[sd_col] * scale
    return out


def _data_driven_ylim(
    strand_data: pd.DataFrame,
    *,
    mean_col: str = "mean_r2_disc",
    sd_col: str = "sd_r2_disc",
    disc_baseline: float | None = None,
) -> tuple[float, float]:
    y_min = float((strand_data[mean_col] - strand_data[sd_col]).min())
    y_max = float((strand_data[mean_col] + strand_data[sd_col]).max())
    y_pad = max(0.004, (y_max - y_min) * 0.12)
    y_lo = y_min - y_pad
    y_hi = y_max + y_pad
    if disc_baseline is not None:
        y_lo = min(y_lo, disc_baseline)
        y_hi = max(y_hi, disc_baseline)
    return y_lo, y_hi


def main() -> None:
    p = argparse.ArgumentParser(
        description="Uncorrected discrepancy-based pseudo-R² figure for Study 1 panels only (appendix)."
    )
    p.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Aggregated table with chance-corrected mean_r2 (default: main raw CSV).",
    )
    p.add_argument(
        "--output-basename",
        type=str,
        default=DEFAULT_OUT_BASE,
        help="Output figure basename under figures/.",
    )
    p.add_argument(
        "--n-per-group",
        type=int,
        default=None,
        help="Sequences per group for the R²_disc affine map (default: read config.json).",
    )
    p.add_argument(
        "--config-json",
        type=Path,
        default=DEFAULT_CONFIG_JSON,
        help="Simulation config with n_sequences_per_group.",
    )
    args = p.parse_args()
    input_csv = args.input_csv
    if not input_csv.is_file():
        raise FileNotFoundError(
            f"Missing input CSV: {input_csv}\n"
            f"Build a full aggregate first (see lcpspell-paper-replication/README.md)."
        )

    ensure_matching_sample_size_source(
        input_path=input_csv,
        default_input_path=DEFAULT_INPUT_CSV,
        config_path=args.config_json,
        default_config_path=DEFAULT_CONFIG_JSON,
        n_per_group=args.n_per_group,
    )

    df = pd.read_csv(input_csv)
    validate_figure_frame(df)

    df = df[df["strand"].isin(STUDY1_STRANDS)].copy()
    missing_strands = set(STUDY1_STRANDS) - set(df["strand"].unique())
    if missing_strands:
        raise ValueError(
            f"Missing Study 1 strands in input CSV: {sorted(missing_strands)}"
        )

    n = resolve_n_total(n_per_group=args.n_per_group, config_path=args.config_json)
    disc_baseline = expected_disc_baseline(n)
    df_disc = to_disc_pseudo_r2(df, n=n)

    n_panels = len(STUDY1_PANEL_SPECS)
    fig, axes = plt.subplots(2, 3, figsize=(28, 12), constrained_layout=True)
    axes = np.array(axes).flatten()

    for idx, spec in enumerate(STUDY1_PANEL_SPECS):
        ax = axes[idx]
        strand_name = spec["strand_label"]
        sd = df_disc[df_disc["strand"] == strand_name].copy()
        if len(sd) == 0:
            raise ValueError(f"No rows for Study 1 strand {strand_name!r}.")

        sd["family"] = sd["method"].apply(get_family)
        sd["family_order"] = sd["family"].map({f: i for i, f in enumerate(FAMILY_ORDER)})
        sd = sd.sort_values(["family_order", "mean_r2_disc"], ascending=[True, False])

        x = np.arange(len(sd))
        for j, row in enumerate(sd.itertuples(index=False)):
            ax.bar(
                x[j],
                row.mean_r2_disc,
                yerr=row.sd_r2_disc,
                color=COLORS.get(row.family, "#9CA3AF"),
                alpha=0.75,
                edgecolor="none",
                capsize=2,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [short_method_label(m) for m in sd["method"]],
            rotation=90,
            ha="center",
            fontsize=8,
        )
        ax.set_ylabel(r"Mean $R^2_{\mathrm{disc}}$")
        ax.set_title(
            f"({spec['panel']}) {spec['short_title']}",
            loc="left",
            fontweight="bold",
        )
        ax.axhline(y=disc_baseline, color="#9CA3AF", linewidth=0.8, linestyle="--", alpha=0.8)
        y_lo, y_hi = _data_driven_ylim(sd, disc_baseline=disc_baseline)
        ax.set_ylim([y_lo, y_hi])
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for j in range(n_panels, len(axes)):
        axes[j].set_visible(False)

    output_pdf = FIGURES_DIR / f"{args.output_basename}.pdf"
    output_png = FIGURES_DIR / f"{args.output_basename}.png"
    plt.savefig(output_pdf, dpi=500, bbox_inches="tight", pad_inches=0.12)
    plt.savefig(output_png, dpi=500, bbox_inches="tight", pad_inches=0.12)
    plt.close()

    print(f"Saved: {output_pdf}")
    print(f"Saved: {output_png}")


if __name__ == "__main__":
    main()
