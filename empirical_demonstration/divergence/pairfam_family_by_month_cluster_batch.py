"""
Batch clustering for Pairfam family-by-month sequence data (paper Section 6).

Runs weighted PAM (``PAMonce``) clustering and sequence index plots for Hamming,
OM, OMspell, OMspellRS, LCP, and LCPspell (expcost 0.5 and 2). Ward linkage
outputs for the appendix are available via ``--clustering-method ward`` (saved under
``ward/`` by default).

Distance settings follow the empirical demonstration in ``main.tex``:
``indel=1``, ``sm='CONSTANT'`` for OM/spell methods; ``duration_ref`` defaults to
the observation window (264 months) for OMspellRS and LCPspell; main-text figure
uses ``norm='none'`` (raw distances). Use ``--norm auto`` for appendix robustness.

Usage::

    python3 pairfam_family_by_month_cluster_batch.py
    python3 pairfam_family_by_month_cluster_batch.py --clustering-method ward
    python3 pairfam_family_by_month_cluster_batch.py --norm auto --clustering-method ward
    python3 .../pairfam_family_by_month_cluster_batch.py --expcosts 0.5 132 --method OMspellRS
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

# Non-interactive backend before Sequenzo imports matplotlib (plot_sequence_index
# calls plt.show() even when save_as is set).
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.ioff()


def _disable_interactive_show() -> None:
    """Sequenzo still calls plt.show() after savefig; make that a no-op in batch runs."""

    def _noop_show(*_args, **_kwargs):
        return None

    plt.show = _noop_show


_disable_interactive_show()

_DEMO_ROOT = Path(__file__).resolve().parent.parent
if str(_DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEMO_ROOT))

from cluster_partition import (  # noqa: E402
    CLUSTERING_METHODS,
    default_output_dir,
    pam_cluster_distribution,
    pam_membership_table,
    ward_partition,
)
from sequenzo import (  # noqa: E402
    get_distance_matrix,
    load_dataset,
    plot_sequence_index,
    SequenceData,
)

_SPELL_EXPCOST_METHODS = frozenset({"OMspell", "OMspellRS", "LCPspell"})
_POSITION_METHODS = frozenset({"HAM", "OM", "LCP"})
_ALL_METHODS = _POSITION_METHODS | _SPELL_EXPCOST_METHODS
_LEGEND_DPI = 400


def load_pairfam_sequence_data() -> SequenceData:
    """Load pairfam family-by-month data and return a sequenzo ``SequenceData`` object."""
    df = load_dataset("pairfam_family_by_month")
    time_list = [f"{i}" for i in range(1, 265)]
    states = list(range(1, 10))
    labels = [
        "Single, no child",
        "Living apart together, no child",
        "Cohabiting, no child",
        "Married, no child",
        "Single, with child(ren)",
        "LAT, with child(ren)",
        "Cohabiting, with child(ren)",
        "Married, 1 child",
        "Married, 2+ children",
    ]
    colors = [
        "#74C9B4",
        "#A6E3D0",
        "#F9E79F",
        "#F6CDA3",
        "#F5B7B1",
        "#D7BDE2",
        "#A3C4F3",
        "#7FB3D5",
        "#EAECEE",
    ]
    return SequenceData(
        df,
        time=time_list,
        id_col="id",
        states=states,
        labels=labels,
        weights=df["weight40"].values,
        custom_colors=colors,
    )


def save_state_legends(sequence_data: SequenceData, output_dir: Path, dpi: int = _LEGEND_DPI) -> None:
    """Export a borderless horizontal state legend as a high-resolution PDF."""
    output_dir.mkdir(parents=True, exist_ok=True)

    legend_path = output_dir / "legend-horizontal.pdf"
    sequence_data.plot_legend(
        save_as=str(legend_path),
        dpi=dpi,
        style="horizontal",
        show_border=False,
    )
    print(f"Saved: {legend_path}")

    plt.close("all")


def _distance_matrix_kwargs(
    method: str,
    expcost: float | None = None,
    norm: str = "none",
) -> dict[str, Any]:
    """Keyword arguments for ``get_distance_matrix`` for one empirical configuration."""
    kwargs: dict[str, Any] = {
        "method": method,
        "norm": norm,
        "full_matrix": True,
    }
    if method in _SPELL_EXPCOST_METHODS:
        if expcost is None:
            raise ValueError(f"expcost is required for method {method!r}")
        kwargs.update(indel=1, sm="CONSTANT", expcost=expcost)
    elif method == "OM":
        kwargs.update(indel=1, sm="CONSTANT")
    elif method not in _POSITION_METHODS:
        raise ValueError(
            f"Unsupported method {method!r}. "
            f"Expected one of {_ALL_METHODS}."
        )
    return kwargs


def _output_basename(method: str, expcost: float | None = None, norm: str = "none") -> str:
    base = f"{method}-expcost-{expcost}".replace(".0", "") if method in _SPELL_EXPCOST_METHODS else (
        "OM-indel-1-sm-CONSTANT" if method == "OM" else method
    )
    if norm != "none":
        return f"{base}-norm-{norm}"
    return base


def run_one_configuration(
    sequence_data: SequenceData,
    method: str,
    output_dir: Path,
    expcost: float | None = None,
    num_clusters: int = 4,
    norm: str = "none",
    show_diagnostic_plots: bool = False,
    clustering_method: str = "pam",
) -> None:
    """Cluster sequences for one dissimilarity setting and save a sequence index plot PDF."""
    label = _output_basename(method, expcost, norm=norm)
    if method in _SPELL_EXPCOST_METHODS:
        print(
            f"\n=== Running {method} "
            f"(indel=1, sm=CONSTANT, expcost={expcost}, norm={norm}, "
            f"clustering={clustering_method}) ==="
        )
    elif method == "OM":
        print(
            f"\n=== Running {method} "
            f"(indel=1, sm=CONSTANT, norm={norm}, clustering={clustering_method}) ==="
        )
    else:
        print(f"\n=== Running {method} (norm={norm}, clustering={clustering_method}) ===")

    dm_kwargs = _distance_matrix_kwargs(method, expcost=expcost, norm=norm)
    distance_matrix = get_distance_matrix(seqdata=sequence_data, **dm_kwargs)
    weights = sequence_data.weights

    if clustering_method == "pam":
        membership_table = pam_membership_table(
            distance_matrix,
            sequence_data.ids,
            weights,
            num_clusters,
        )
        distribution = pam_cluster_distribution(membership_table, weights)
    else:
        membership_table, distribution, _ = ward_partition(
            distance_matrix,
            sequence_data.ids,
            weights,
            num_clusters,
            show_diagnostic_plots=show_diagnostic_plots,
        )

    print(membership_table)
    print(distribution)

    output_path = output_dir / f"{label}.pdf"
    try:
        plot_sequence_index(
            seqdata=sequence_data,
            group_dataframe=membership_table,
            group_column_name="Cluster",
            nrows=1,
            ncols=4,
            include_legend=False,
            show_sequence_ids=False,
            save_as=str(output_path),
            dpi=400,
        )
    finally:
        plt.close("all")

    print(f"Saved: {output_path}")


def run_all(
    sequence_data: SequenceData,
    output_dir: str | Path = ".",
    expcosts: list[float] | None = None,
    norm: str = "none",
    methods: list[str] | None = None,
    show_diagnostic_plots: bool = False,
    num_clusters: int = 4,
    clustering_method: str = "pam",
) -> None:
    """
    Run Section 6 dissimilarity measures and save PDFs under ``output_dir``.

    Default order: HAM, OM, OMspell / OMspellRS / LCPspell (each expcost), LCP.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if expcosts is None:
        expcosts = [0.5, 2.0]

    if methods is None:
        run_one_configuration(
            sequence_data, "HAM", output_dir,
            norm=norm, num_clusters=num_clusters,
            show_diagnostic_plots=show_diagnostic_plots,
            clustering_method=clustering_method,
        )
        run_one_configuration(
            sequence_data, "OM", output_dir,
            norm=norm, num_clusters=num_clusters,
            show_diagnostic_plots=show_diagnostic_plots,
            clustering_method=clustering_method,
        )
        for spell_method in ("OMspell", "OMspellRS", "LCPspell"):
            for expcost in expcosts:
                run_one_configuration(
                    sequence_data,
                    spell_method,
                    output_dir,
                    expcost=expcost,
                    norm=norm,
                    num_clusters=num_clusters,
                    show_diagnostic_plots=show_diagnostic_plots,
                    clustering_method=clustering_method,
                )
        run_one_configuration(
            sequence_data, "LCP", output_dir,
            norm=norm, num_clusters=num_clusters,
            show_diagnostic_plots=show_diagnostic_plots,
            clustering_method=clustering_method,
        )
        return

    for method in methods:
        if method in _SPELL_EXPCOST_METHODS:
            for expcost in expcosts:
                run_one_configuration(
                    sequence_data, method, output_dir,
                    expcost=expcost, norm=norm, num_clusters=num_clusters,
                    show_diagnostic_plots=show_diagnostic_plots,
                    clustering_method=clustering_method,
                )
        else:
            run_one_configuration(
                sequence_data, method, output_dir,
                norm=norm, num_clusters=num_clusters,
                show_diagnostic_plots=show_diagnostic_plots,
                clustering_method=clustering_method,
            )


def _parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Pairfam family-by-month clustering batch (paper Section 6)."
    )
    parser.add_argument(
        "--clustering-method",
        choices=CLUSTERING_METHODS,
        default="pam",
        help="Partition rule: weighted PAM (main text) or Ward linkage (appendix).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for per-method PDF outputs. "
            "Default: this folder for PAM; ward/ (or ward/auto_norm/) for Ward."
        ),
    )
    parser.add_argument(
        "--norm",
        default="none",
        help='Normalization for get_distance_matrix (default: none; use "auto" for appendix).',
    )
    parser.add_argument(
        "--method",
        action="append",
        dest="methods",
        choices=sorted(_ALL_METHODS),
        help="Run only this method (repeatable). Default: all Section 6 methods.",
    )
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=4,
        help="Fixed number of clusters for comparability across methods (default: 4).",
    )
    parser.add_argument(
        "--show-diagnostic-plots",
        action="store_true",
        help=(
            "Compute dendrogram / CQI diagnostics (printed tables only; "
            "headless backend; no GUI windows)."
        ),
    )
    parser.add_argument(
        "--expcosts",
        type=float,
        nargs="+",
        default=[0.5, 2.0],
        help="expcost grid for spell-based methods (default: 0.5 2).",
    )
    parser.add_argument(
        "--legends-only",
        action="store_true",
        help="Only export legend-horizontal.pdf.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    here = Path(__file__).resolve().parent
    if args.output_dir is None:
        args.output_dir = default_output_dir(here, args.clustering_method, args.norm)
    print("Loading pairfam_family_by_month …")
    sequence_data = load_pairfam_sequence_data()
    print(f"Output directory: {args.output_dir.resolve()}")
    save_state_legends(sequence_data, here)
    if args.legends_only:
        print("\nLegend export finished.")
        return
    print(
        f"clustering={args.clustering_method!r}, norm={args.norm!r}, "
        f"expcost grid: {args.expcosts}, k={args.num_clusters}"
    )
    if args.methods:
        print(f"methods: {args.methods}")
    run_all(
        sequence_data,
        output_dir=args.output_dir,
        expcosts=args.expcosts,
        norm=args.norm,
        methods=args.methods,
        show_diagnostic_plots=args.show_diagnostic_plots,
        num_clusters=args.num_clusters,
        clustering_method=args.clustering_method,
    )
    print("\nAll configurations finished.")


if __name__ == "__main__":
    main()
