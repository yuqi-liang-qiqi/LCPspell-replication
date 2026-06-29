"""
Partition helpers for empirical demonstration batch scripts.

Main-text figures use weighted PAM (``PAMonce`` via :func:`KMedoids`).
Appendix Ward linkage results are written under a ``ward/`` subfolder.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sequenzo import Cluster, ClusterQuality, ClusterResults
from sequenzo.clustering import KMedoids, cluster_labels_from_kmedoids_result

CLUSTERING_METHODS = ("pam", "ward")


def default_output_dir(script_dir: Path, clustering_method: str, norm: str) -> Path:
    """Return the default PDF output directory for a clustering / norm setting."""
    if clustering_method not in CLUSTERING_METHODS:
        raise ValueError(
            f"Unsupported clustering_method {clustering_method!r}. "
            f"Expected one of {CLUSTERING_METHODS}."
        )
    base = script_dir if clustering_method == "pam" else script_dir / "ward"
    if norm != "none":
        return base / "auto_norm"
    return base


def pam_membership_table(
    distance_matrix: np.ndarray,
    entity_ids,
    weights: np.ndarray,
    num_clusters: int,
) -> pd.DataFrame:
    """Weighted PAM partition mapped to Sequenzo-style 1-based cluster ids."""
    dm = np.asarray(distance_matrix, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    raw = KMedoids(dm, k=num_clusters, weights=w, method="PAMonce", verbose=False)
    labels = cluster_labels_from_kmedoids_result(raw).astype(int) + 1
    return pd.DataFrame({"Entity ID": entity_ids, "Cluster": labels})


def pam_cluster_distribution(
    membership_table: pd.DataFrame,
    weights: np.ndarray,
) -> pd.DataFrame:
    """Unweighted and weighted cluster shares for a PAM partition."""
    w = np.asarray(weights, dtype=np.float64)
    entity_ids = membership_table["Entity ID"].to_numpy()
    id_to_weight = dict(zip(entity_ids, w, strict=True))
    rows: list[dict[str, float | int]] = []
    total_n = len(membership_table)
    total_w = float(w.sum())
    for cluster_id, group in membership_table.groupby("Cluster", sort=True):
        count = len(group)
        weight_sum = float(sum(id_to_weight[eid] for eid in group["Entity ID"]))
        rows.append(
            {
                "Cluster": int(cluster_id),
                "Count": count,
                "Percentage": round(100.0 * count / total_n, 2),
                "Weight_Sum": weight_sum,
                "Weight_Percentage": round(100.0 * weight_sum / total_w, 2),
            }
        )
    return pd.DataFrame(rows)


def ward_partition(
    distance_matrix: np.ndarray,
    entity_ids,
    weights: np.ndarray,
    num_clusters: int,
    show_diagnostic_plots: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, ClusterQuality]:
    """Ward hierarchical partition with optional dendrogram / CQI diagnostics."""
    cluster = Cluster(
        distance_matrix,
        entity_ids,
        clustering_method="ward_d",
        weights=weights,
    )
    cluster_quality = ClusterQuality(cluster)
    cluster_quality.compute_cluster_quality_scores()
    print(cluster_quality.get_cqi_table())

    if show_diagnostic_plots:
        cluster.plot_dendrogram(xlabel="Sequences", ylabel="Distance")
        cluster_quality.plot_cqi_scores(norm="zscore")

    cluster_results = ClusterResults(cluster)
    membership_table = cluster_results.get_cluster_memberships(num_clusters=num_clusters)
    distribution = cluster_results.get_cluster_distribution(num_clusters=num_clusters)
    if show_diagnostic_plots:
        cluster_results.plot_cluster_distribution(num_clusters=num_clusters, title=None)
    return membership_table, distribution, cluster_quality
