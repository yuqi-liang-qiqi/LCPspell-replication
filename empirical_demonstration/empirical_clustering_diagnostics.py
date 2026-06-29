"""
Cluster-quality diagnostics and linkage robustness for empirical demonstrations.

Computes weighted CQI (PBC, ASWw, HC, R2) at fixed k=4 for Ward and PAM partitions,
plus ASWw under Ward, average linkage, and PAM for linkage-robustness tables.

Outputs JSON summaries and LaTeX tables under ``lcpspell-paper-replication/tables/``.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import fcluster

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPLICATION_ROOT = _SCRIPT_DIR.parent
_TABLES_DIR = _REPLICATION_ROOT / "tables"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from divergence.pairfam_family_by_month_cluster_batch import (  # noqa: E402
    _distance_matrix_kwargs as pairfam_dm_kwargs,
    _output_basename as pairfam_basename,
    load_pairfam_sequence_data,
)
from convergence.mvad_school_to_work_cluster_batch import (  # noqa: E402
    _distance_matrix_kwargs as mvad_dm_kwargs,
    _output_basename as mvad_basename,
    load_mvad_sequence_data,
)

from sequenzo import Cluster, get_distance_matrix  # noqa: E402
from sequenzo.clustering import KMedoids, cluster_labels_from_kmedoids_result  # noqa: E402
from sequenzo.clustering.validation.partition_quality import (  # noqa: E402
    compute_partition_quality,
)

K_FIXED = 4
CQI_METRICS = ("PBC", "ASWw", "HC", "R2")
LINKAGE_METHODS = ("ward_d", "average", "pam")
LINKAGE_LABELS = {
    "ward_d": "Ward",
    "average": "Average",
    "pam": "PAM",
}


@dataclass(frozen=True)
class MethodConfig:
    method: str
    expcost: float | None = None

    @property
    def label(self) -> str:
        if self.expcost is not None:
            return f"{self.method} ({self.expcost:g})"
        return self.method


PAIRFAM_METHODS = [
    MethodConfig("HAM"),
    MethodConfig("OM"),
    MethodConfig("OMspell", 0.5),
    MethodConfig("OMspell", 2.0),
    MethodConfig("OMspellRS", 0.5),
    MethodConfig("OMspellRS", 2.0),
    MethodConfig("LCP"),
    MethodConfig("LCPspell", 0.5),
    MethodConfig("LCPspell", 2.0),
]

MVAD_METHODS = [
    MethodConfig("HAM"),
    MethodConfig("OM"),
    MethodConfig("OMspell", 0.5),
    MethodConfig("OMspell", 2.0),
    MethodConfig("OMspellRS", 0.5),
    MethodConfig("OMspellRS", 2.0),
    MethodConfig("RLCP"),
    MethodConfig("RLCPspell", 0.5),
    MethodConfig("RLCPspell", 2.0),
]


def _weights(seqdata) -> np.ndarray:
    w = np.asarray(seqdata.weights, dtype=np.float64)
    if w.shape[0] != seqdata.n_sequences:
        return np.ones(seqdata.n_sequences, dtype=np.float64)
    return w


def _distance_matrix(seqdata, dm_kwargs_fn, cfg: MethodConfig, norm: str = "none"):
    kwargs = dm_kwargs_fn(cfg.method, expcost=cfg.expcost, norm=norm)
    return get_distance_matrix(seqdata=seqdata, **kwargs)


def _ward_labels(cluster: Cluster, k: int) -> np.ndarray:
    return fcluster(cluster.linkage_matrix, t=k, criterion="maxclust").astype(int)


def _pam_labels(distance_matrix: np.ndarray, k: int, weights: np.ndarray) -> np.ndarray:
    dm = np.asarray(distance_matrix, dtype=np.float64)
    raw = KMedoids(dm, k=k, weights=weights, method="PAMonce", verbose=False)
    return cluster_labels_from_kmedoids_result(raw).astype(int)


def _labels_for_linkage(
    distance_matrix: np.ndarray,
    entity_ids,
    weights: np.ndarray,
    linkage: str,
    k: int,
) -> np.ndarray:
    if linkage == "pam":
        return _pam_labels(distance_matrix, k, weights)
    cluster = Cluster(
        distance_matrix,
        entity_ids,
        clustering_method=linkage,
        weights=weights,
    )
    return _ward_labels(cluster, k)


def _cqi_for_partition(
    distance_matrix: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    scores = compute_partition_quality(distance_matrix, labels, weights)
    return {metric: float(scores[metric]) for metric in CQI_METRICS}


def _asww_at_k(
    distance_matrix: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> float:
    return float(
        compute_partition_quality(distance_matrix, labels, weights)["ASWw"]
    )


def analyse_configuration(
    seqdata,
    dm_kwargs_fn,
    basename_fn,
    cfg: MethodConfig,
    norm: str = "none",
) -> dict[str, Any]:
    dm = _distance_matrix(seqdata, dm_kwargs_fn, cfg, norm=norm)
    weights = _weights(seqdata)

    ward_labels = _labels_for_linkage(
        dm, seqdata.ids, weights, "ward_d", K_FIXED
    )
    pam_labels = _labels_for_linkage(dm, seqdata.ids, weights, "pam", K_FIXED)
    cqi_at_k4 = {
        "ward_d": _cqi_for_partition(dm, ward_labels, weights),
        "pam": _cqi_for_partition(dm, pam_labels, weights),
    }

    asww_linkage: dict[str, float] = {}
    for linkage in LINKAGE_METHODS:
        labels = _labels_for_linkage(dm, seqdata.ids, weights, linkage, K_FIXED)
        asww_linkage[linkage] = _asww_at_k(dm, labels, weights)

    return {
        "config_id": basename_fn(cfg.method, cfg.expcost, norm=norm),
        "label": cfg.label,
        "cqi_at_k4": cqi_at_k4,
        "asww_linkage": asww_linkage,
        "asww_pam": asww_linkage["pam"],
    }


def analyse_dataset(
    name: str,
    seqdata,
    dm_kwargs_fn,
    basename_fn,
    methods: list[MethodConfig],
) -> list[dict[str, Any]]:
    results = []
    for cfg in methods:
        print(f"[{name}] {cfg.label} …")
        results.append(
            analyse_configuration(seqdata, dm_kwargs_fn, basename_fn, cfg)
        )
    return results


def _fmt(x: float, digits: int = 3) -> str:
    if not np.isfinite(x):
        return "---"
    return f"{x:.{digits}f}"


def _latex_cqi_row_cells(cqi: dict[str, float]) -> str:
    return (
        f"{_fmt(cqi['PBC'])} & {_fmt(cqi['ASWw'])} & {_fmt(cqi['HC'])} & "
        f"{_fmt(cqi['R2'])}"
    )


def _latex_cqi_table(
    rows: list[dict[str, Any]],
    caption: str,
    label: str,
) -> str:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{l*{8}{r}}",
        "\\toprule",
        "& \\multicolumn{4}{c}{PAM ($k=4$)} & \\multicolumn{4}{c}{Ward ($k=4$)} \\\\",
        "\\cmidrule(lr){2-5} \\cmidrule(lr){6-9}",
        "Measure & PBC & ASWw & HC & $R^2$ & PBC & ASWw & HC & $R^2$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        cqi = row["cqi_at_k4"]
        lines.append(
            f"{row['label']} & "
            f"{_latex_cqi_row_cells(cqi['pam'])} & "
            f"{_latex_cqi_row_cells(cqi['ward_d'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


CROSS_METHOD_PAIRS = {
    "pairfam": [
        ("HAM", "OM"),
        ("OM", "LCP"),
        ("OM", "LCPspell (0.5)"),
        ("LCP", "LCPspell (0.5)"),
        ("OMspell (0.5)", "OMspellRS (0.5)"),
    ],
    "mvad": [
        ("HAM", "OM"),
        ("OM", "RLCP"),
        ("OM", "RLCPspell (0.5)"),
        ("RLCP", "RLCPspell (0.5)"),
        ("OMspellRS (0.5)", "RLCPspell (0.5)"),
    ],
}


def _cross_method_asww_rows(
    configs: list[dict[str, Any]],
    pairs: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    by_label = {row["label"]: row["asww_pam"] for row in configs}
    out = []
    for left, right in pairs:
        out.append(
            {
                "left": left,
                "right": right,
                "left_asww": by_label[left],
                "right_asww": by_label[right],
            }
        )
    return out


def _latex_cross_method_asww_table(
    rows: list[dict[str, Any]],
    caption: str,
    label: str,
) -> str:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\footnotesize",
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "Comparison (PAM, $k=4$) & ASWw (left) & ASWw (right) \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['left']} vs.\\ {row['right']} & "
            f"{_fmt(row['left_asww'])} & {_fmt(row['right_asww'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def _latex_asww_linkage_table(
    rows: list[dict[str, Any]],
    caption: str,
    label: str,
) -> str:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\footnotesize",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Measure & Ward ASWw & Average ASWw & PAM ASWw \\\\",
        "\\midrule",
    ]
    for row in rows:
        asww = row["asww_linkage"]
        lines.append(
            f"{row['label']} & "
            f"{_fmt(asww['ward_d'])} & "
            f"{_fmt(asww['average'])} & "
            f"{_fmt(asww['pam'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    _TABLES_DIR.mkdir(parents=True, exist_ok=True)

    pairfam = load_pairfam_sequence_data()
    pairfam_rows = analyse_dataset(
        "pairfam", pairfam, pairfam_dm_kwargs, pairfam_basename, PAIRFAM_METHODS
    )

    mvad = load_mvad_sequence_data()
    mvad_rows = analyse_dataset(
        "mvad", mvad, mvad_dm_kwargs, mvad_basename, MVAD_METHODS
    )

    cross_pairfam = _cross_method_asww_rows(pairfam_rows, CROSS_METHOD_PAIRS["pairfam"])
    cross_mvad = _cross_method_asww_rows(mvad_rows, CROSS_METHOD_PAIRS["mvad"])

    out = {
        "pairfam": pairfam_rows,
        "mvad": mvad_rows,
        "cross_method_asww": {"pairfam": cross_pairfam, "mvad": cross_mvad},
        "k_fixed": K_FIXED,
    }
    json_path = _TABLES_DIR / "empirical_clustering_diagnostics.json"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {json_path}")

    pairfam_cqi_tex = _latex_cqi_table(
        pairfam_rows,
        "Divergence demonstration: cluster-quality indicators at $k=4$ under "
        "weighted PAM and Ward linkage. Metrics use sequence weights in the "
        "quality evaluation; Ward linkage itself is unweighted. PAM results "
        "appear in the left block and Ward results on the right.",
        "tab:empirical-cqi-pairfam",
    )
    pairfam_asww_tex = _latex_asww_linkage_table(
        pairfam_rows,
        "Divergence demonstration: linkage robustness at $k=4$, measured by "
        "weighted average silhouette width (ASWw) under Ward, average linkage, "
        "and weighted PAM partitions.",
        "tab:empirical-asww-linkage-pairfam",
    )
    mvad_cqi_tex = _latex_cqi_table(
        mvad_rows,
        "Convergence demonstration: cluster-quality indicators at $k=4$ under "
        "weighted PAM and Ward linkage. Metrics use sequence weights in the "
        "quality evaluation; Ward linkage itself is unweighted. PAM results "
        "appear in the left block and Ward results on the right.",
        "tab:empirical-cqi-mvad",
    )
    mvad_asww_tex = _latex_asww_linkage_table(
        mvad_rows,
        "Convergence demonstration: linkage robustness at $k=4$, measured by "
        "weighted average silhouette width (ASWw) under Ward, average linkage, "
        "and weighted PAM partitions.",
        "tab:empirical-asww-linkage-mvad",
    )

    combined = (
        "% Auto-generated by empirical_clustering_diagnostics.py\n"
        + pairfam_cqi_tex
        + pairfam_asww_tex
        + mvad_cqi_tex
        + mvad_asww_tex
    )
    cluster_quality = (
        "% Auto-generated by empirical_clustering_diagnostics.py\n"
        + pairfam_cqi_tex
        + mvad_cqi_tex
    )
    linkage_robustness = (
        "% Auto-generated by empirical_clustering_diagnostics.py\n"
        + pairfam_asww_tex
        + mvad_asww_tex
    )
    tex_path = _TABLES_DIR / "tab_empirical_clustering_diagnostics.tex"
    tex_path.write_text(combined, encoding="utf-8")
    print(f"Wrote {tex_path}")
    (_TABLES_DIR / "tab_empirical_cluster_quality.tex").write_text(
        cluster_quality, encoding="utf-8"
    )
    print(f"Wrote {_TABLES_DIR / 'tab_empirical_cluster_quality.tex'}")
    (_TABLES_DIR / "tab_empirical_linkage_robustness.tex").write_text(
        linkage_robustness, encoding="utf-8"
    )
    print(f"Wrote {_TABLES_DIR / 'tab_empirical_linkage_robustness.tex'}")


if __name__ == "__main__":
    main()
