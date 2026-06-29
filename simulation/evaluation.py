"""
@Author  : Yuqi Liang 梁彧祺
@File    : evaluation.py
@Time    : 19/01/2026 10:10
@Desc    :
Evaluation module for sensitivity analysis.

Computes the conventional discrepancy-based pseudo-R² (R²_disc; Studer et al.,
2011) and a balanced two-group finite-sample chance-corrected variant
(R²_cc) for the main simulations.
"""

import logging
from typing import Dict, List, Literal, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DirectionalContrast = Literal["divergence", "convergence"]


def compute_total_discrepancy(distance_matrix: np.ndarray) -> float:
    """Sum of upper-triangle pairwise dissimilarities."""
    n = distance_matrix.shape[0]
    return float(np.sum(distance_matrix[np.triu_indices(n, k=1)]))


def _validate_binary_group_inputs(
    distance_matrix: np.ndarray,
    group_labels: np.ndarray,
) -> Tuple[int, int, int]:
    """
    Validate inputs for balanced binary-group pseudo-R² (labels 0 and 1 only).

    Returns (n, n0, n1).
    """
    if distance_matrix.ndim != 2:
        raise ValueError(
            f"distance_matrix must be 2-D, got shape {distance_matrix.shape}"
        )
    n, m = distance_matrix.shape
    if n != m:
        raise ValueError(f"distance_matrix must be square, got {n} x {m}")

    labels = np.asarray(group_labels)
    if labels.ndim != 1:
        raise ValueError(
            f"group_labels must be one-dimensional, got shape {labels.shape}."
        )
    if labels.shape[0] != n:
        raise ValueError(
            f"group_labels length {labels.shape[0]} != matrix size {n}"
        )

    dist = np.asarray(distance_matrix, dtype=np.float64)
    if not np.all(np.isfinite(dist)):
        raise ValueError("distance_matrix contains non-finite values")

    if not np.allclose(dist, dist.T, rtol=0, atol=1e-8):
        max_asymmetry = float(np.max(np.abs(dist - dist.T)))
        raise ValueError(
            "distance_matrix must be symmetric. "
            f"Maximum asymmetry: {max_asymmetry:.3e}"
        )

    if not np.allclose(np.diag(dist), 0.0, rtol=0, atol=1e-8):
        raise ValueError("distance_matrix diagonal must be approximately zero.")

    if float(np.min(dist)) < -1e-8:
        raise ValueError(
            "distance_matrix contains negative values. "
            "The main chance-corrected score assumes non-negative dissimilarities."
        )

    unique_groups = np.unique(labels)
    if not np.array_equal(unique_groups, np.array([0, 1])):
        raise ValueError(
            "compute_pseudo_r2 currently requires exactly two groups "
            "encoded as 0 and 1."
        )

    n0 = int(np.sum(labels == 0))
    n1 = int(np.sum(labels == 1))
    if n0 == 0 or n1 == 0:
        raise ValueError("Both groups must be non-empty.")

    if n0 != n1:
        raise ValueError(
            "compute_pseudo_r2 uses the balanced two-group simulation formula. "
            f"Got unequal group sizes: n0={n0}, n1={n1}."
        )

    return n, n0, n1


def compute_between_group_discrepancy(
    distance_matrix: np.ndarray,
    group_labels: np.ndarray,
) -> float:
    """Sum of distances between sequences in group 0 and group 1."""
    group0_indices = np.flatnonzero(group_labels == 0)
    group1_indices = np.flatnonzero(group_labels == 1)
    return float(
        np.sum(distance_matrix[np.ix_(group0_indices, group1_indices)])
    )


def compute_within_group_discrepancy(
    distance_matrix: np.ndarray,
    group_labels: np.ndarray,
) -> Tuple[float, float]:
    """Within-group distance sums for groups 0 and 1 (upper triangles)."""
    group0_indices = np.flatnonzero(group_labels == 0)
    group1_indices = np.flatnonzero(group_labels == 1)

    d0 = distance_matrix[np.ix_(group0_indices, group0_indices)]
    d1 = distance_matrix[np.ix_(group1_indices, group1_indices)]

    within0 = float(np.sum(d0[np.triu_indices(len(group0_indices), k=1)]))
    within1 = float(np.sum(d1[np.triu_indices(len(group1_indices), k=1)]))
    return within0, within1


def expected_disc_baseline(n: int) -> float:
    """Expected R²_disc under random labeling (unweighted, two groups)."""
    if n <= 1:
        return 0.0
    return 1.0 / (n - 1)


def chance_corrected_affine_scale(n: int) -> float:
    """Scale 1 - 1/(n-1) for mapping R²_disc to R²_cc."""
    return 1.0 - expected_disc_baseline(n)


def disc_from_chance_corrected(r2_cc: float, n: int) -> float:
    """Map chance-corrected pseudo-R² (R²_cc) to uncorrected R²_disc."""
    baseline = expected_disc_baseline(n)
    return baseline + chance_corrected_affine_scale(n) * r2_cc


def chance_corrected_from_disc(r2_disc: float, n: int) -> float:
    """Map uncorrected R²_disc to chance-corrected pseudo-R² (R²_cc)."""
    baseline = expected_disc_baseline(n)
    scale = chance_corrected_affine_scale(n)
    if scale <= 0:
        return 0.0
    return (r2_disc - baseline) / scale


def compute_disc_pseudo_r2(
    distance_matrix: np.ndarray,
    group_labels: np.ndarray,
) -> float:
    """
    Uncorrected discrepancy-based pseudo-R² (ν = 1; Studer et al., 2011).

        R²_disc = 1 - SS_W / SS_T

    Requires a balanced two-group partition encoded as 0 and 1.
    """
    n, n0, n1 = _validate_binary_group_inputs(distance_matrix, group_labels)
    dist = np.asarray(distance_matrix, dtype=np.float64)

    total_sum = float(np.sum(dist[np.triu_indices(n, k=1)]))
    if np.isclose(total_sum, 0.0):
        return 0.0

    group0 = np.flatnonzero(group_labels == 0)
    group1 = np.flatnonzero(group_labels == 1)

    d0 = dist[np.ix_(group0, group0)]
    d1 = dist[np.ix_(group1, group1)]

    within0 = float(np.sum(d0[np.triu_indices(n0, k=1)]))
    within1 = float(np.sum(d1[np.triu_indices(n1, k=1)]))

    ss_total = total_sum / n
    ss_within = within0 / n0 + within1 / n1
    return float(1.0 - ss_within / ss_total)


def compute_pseudo_r2(
    distance_matrix: np.ndarray,
    group_labels: np.ndarray,
) -> float:
    """
    Balanced two-group finite-sample chance-corrected pseudo-R² (main metric).

    For n0 = n1 = m and n = 2m this equals
    (R²_disc - 1/(n-1)) / (1 - 1/(n-1)), equivalently the q-based formula
    with expected between-pair fraction m²/C(n,2).

    Values are not clipped. Negative scores are valid when separation is
    weaker than expected under random labeling.
    """
    n, n0, n1 = _validate_binary_group_inputs(distance_matrix, group_labels)

    total_discrepancy = compute_total_discrepancy(distance_matrix)
    if np.isclose(total_discrepancy, 0.0):
        return 0.0

    between_discrepancy = compute_between_group_discrepancy(
        distance_matrix, group_labels
    )

    total_pairs = n * (n - 1) / 2
    expected_between_fraction = (n0 * n1) / total_pairs
    observed_between_fraction = between_discrepancy / total_discrepancy

    if expected_between_fraction >= 1.0:
        return 0.0

    pseudo_r2 = (observed_between_fraction - expected_between_fraction) / (
        1.0 - expected_between_fraction
    )

    theoretical_lower_bound = (
        -expected_between_fraction / (1.0 - expected_between_fraction)
    )
    if pseudo_r2 > 1.0 + 1e-8 or pseudo_r2 < theoretical_lower_bound - 1e-8:
        logger.warning(
            "pseudo-R² outside its theoretical range [%.6f, 1]: %.6f "
            "(n=%d, n0=%d, n1=%d)",
            theoretical_lower_bound,
            pseudo_r2,
            n,
            n0,
            n1,
        )

    return pseudo_r2


def standardize_pseudo_r2(
    pseudo_r2_dict: Dict[str, float],
) -> Dict[str, float]:
    """Z-score pseudo-R² values across methods within one strand."""
    values = np.array(list(pseudo_r2_dict.values()))
    mean_val = np.mean(values)
    std_val = np.std(values)

    if std_val == 0:
        return {method: 0.0 for method in pseudo_r2_dict.keys()}

    return {
        method: (pseudo_r2_dict[method] - mean_val) / std_val
        for method in pseudo_r2_dict.keys()
    }


def evaluate_simulation_strand(
    distance_matrices: Dict[str, np.ndarray],
    group_labels: np.ndarray,
) -> Dict[str, float]:
    """Compute chance-corrected pseudo-R² for each distance method."""
    return {
        method: compute_pseudo_r2(dist_matrix, group_labels)
        for method, dist_matrix in distance_matrices.items()
    }


def compute_pseudo_r2_single(
    distance_matrix: np.ndarray,
    group_labels: np.ndarray,
) -> float:
    """Convenience wrapper for a single distance matrix."""
    return compute_pseudo_r2(distance_matrix, group_labels)


def compute_normalized_paired_contrast(
    mean_larger: float,
    mean_smaller: float,
) -> float:
    """
    Scale-invariant directional sensitivity score in [-1, 1].

    S = (mean_larger - mean_smaller) / (mean_larger + mean_smaller).
    Returns 0 when both means are zero.
    """
    denom = mean_larger + mean_smaller
    if np.isclose(denom, 0.0):
        return 0.0
    return float((mean_larger - mean_smaller) / denom)


def compute_paired_win_rate(deltas: np.ndarray) -> float:
    """
    Chance-centered draw-level superiority rate in [-1, 1].

    For nested matched draws, ``deltas`` contains one contrast per draw
    (e.g. ``d_early - d_late``); ``S_win`` summarizes how often the expected
    direction holds across draws, not a classical paired comparison on
    latent units.

    S_win = 2 * P(delta > 0) + P(delta = 0) - 1.
    """
    deltas = np.asarray(deltas, dtype=np.float64)
    if deltas.size == 0:
        return 0.0
    ties = np.isclose(deltas, 0.0, rtol=0.0, atol=1e-10)
    wins = (deltas > 0.0) & ~ties
    score = float(2.0 * (np.mean(wins) + 0.5 * np.mean(ties)) - 1.0)
    if score < -1.0 - 1e-8 or score > 1.0 + 1e-8:
        raise RuntimeError(
            f"Paired win rate outside [-1, 1]: {score} (n={deltas.size})."
        )
    return score


def _validate_directional_distance_arrays(
    d_control: np.ndarray,
    d_signal: np.ndarray,
    *,
    label_control: str = "d_control",
    label_signal: str = "d_signal",
) -> None:
    if not np.all(np.isfinite(d_control)):
        raise ValueError(f"{label_control} contains non-finite values.")
    if not np.all(np.isfinite(d_signal)):
        raise ValueError(f"{label_signal} contains non-finite values.")
    if float(np.min(d_control)) < -1e-8 or float(np.min(d_signal)) < -1e-8:
        raise ValueError("Directional scores require non-negative dissimilarities.")


def compute_early_late_directional_scores(
    d_early: np.ndarray,
    d_late: np.ndarray,
    contrast: DirectionalContrast,
) -> Tuple[float, float]:
    """
    Early-vs-late directional sensitivity from matched focal pair distances.

    Each draw contributes within-pair distances for early- and late-placed focal
    directional patterns. Divergence expects d_early > d_late; convergence
    expects d_early < d_late. The contrast uses replication means (not per-draw
    normalized ratios).
    """
    d_early = np.asarray(d_early, dtype=np.float64)
    d_late = np.asarray(d_late, dtype=np.float64)
    if d_early.shape != d_late.shape:
        raise ValueError("d_early and d_late must have the same shape.")
    if d_early.size == 0:
        return 0.0, 0.0

    _validate_directional_distance_arrays(
        d_early, d_late, label_control="d_early", label_signal="d_late"
    )

    if contrast == "divergence":
        deltas = d_early - d_late
        contrast_score = compute_normalized_paired_contrast(
            float(np.mean(d_early)),
            float(np.mean(d_late)),
        )
    elif contrast == "convergence":
        deltas = d_late - d_early
        contrast_score = compute_normalized_paired_contrast(
            float(np.mean(d_late)),
            float(np.mean(d_early)),
        )
    else:
        raise ValueError(
            f"contrast must be 'divergence' or 'convergence', got {contrast!r}."
        )

    if contrast_score < -1.0 - 1e-8 or contrast_score > 1.0 + 1e-8:
        raise RuntimeError(
            f"Normalized paired contrast outside [-1, 1]: {contrast_score}."
        )

    return contrast_score, compute_paired_win_rate(deltas)


def compute_directional_pair_scores(
    d_control: np.ndarray,
    d_signal: np.ndarray,
    contrast: DirectionalContrast,
) -> Tuple[float, float]:
    """
    Aggregate pair-level distances into normalized contrast and win rate.

    Low-level helper with explicit control/signal roles. Prefer
    :func:`compute_early_late_directional_scores` for Study~2 early-vs-late
    panels. Divergence: expect d_signal > d_control. Convergence: expect
    d_signal < d_control.
    """
    d_control = np.asarray(d_control, dtype=np.float64)
    d_signal = np.asarray(d_signal, dtype=np.float64)
    if d_control.shape != d_signal.shape:
        raise ValueError("d_control and d_signal must have the same shape.")
    if d_control.size == 0:
        return 0.0, 0.0

    _validate_directional_distance_arrays(d_control, d_signal)

    if contrast == "divergence":
        deltas = d_signal - d_control
        mean_larger = float(np.mean(d_signal))
        mean_smaller = float(np.mean(d_control))
    elif contrast == "convergence":
        deltas = d_control - d_signal
        mean_larger = float(np.mean(d_control))
        mean_smaller = float(np.mean(d_signal))
    else:
        raise ValueError(
            f"contrast must be 'divergence' or 'convergence', got {contrast!r}."
        )

    contrast_score = compute_normalized_paired_contrast(mean_larger, mean_smaller)
    if contrast_score < -1.0 - 1e-8 or contrast_score > 1.0 + 1e-8:
        raise RuntimeError(
            f"Normalized paired contrast outside [-1, 1]: {contrast_score}."
        )

    return contrast_score, compute_paired_win_rate(deltas)


def aggregate_directional_replications(
    replication_results: List[Dict[str, Dict[str, float]]],
) -> Dict[str, Dict[str, float]]:
    """
    Mean and standard deviation of pair-level scores across replications.

    Each replication entry maps method -> {"contrast": float, "win": float}.
    Output uses ``mean``/``std`` for contrast and ``win_mean``/``win_std`` for win rate.
    """
    if not replication_results:
        return {}

    expected_methods = list(replication_results[0].keys())
    expected_set = set(expected_methods)

    for rep_idx, result in enumerate(replication_results):
        if set(result) != expected_set:
            raise RuntimeError(
                f"Replication {rep_idx} has inconsistent method keys."
            )
        for method, scores in result.items():
            if set(scores) != {"contrast", "win"}:
                raise RuntimeError(
                    f"Replication {rep_idx}, method '{method}' "
                    f"has unexpected score keys: {sorted(scores)}."
                )
            for key, value in scores.items():
                if not np.isfinite(value):
                    raise RuntimeError(
                        f"Replication {rep_idx}, method '{method}', "
                        f"score '{key}' is non-finite: {value}."
                    )

    aggregated: Dict[str, Dict[str, float]] = {}
    for method in expected_methods:
        contrasts = [r[method]["contrast"] for r in replication_results]
        wins = [r[method]["win"] for r in replication_results]
        aggregated[method] = {
            "mean": float(np.mean(contrasts)),
            "std": (
                float(np.std(contrasts, ddof=1))
                if len(contrasts) > 1
                else 0.0
            ),
            "win_mean": float(np.mean(wins)),
            "win_std": (
                float(np.std(wins, ddof=1)) if len(wins) > 1 else 0.0
            ),
        }
    return aggregated


def aggregate_replications(
    replication_results: List[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Mean and standard deviation of pseudo-R² across replications."""
    if not replication_results:
        return {}

    expected_methods = list(replication_results[0].keys())
    expected_set = set(expected_methods)

    for rep_idx, result in enumerate(replication_results):
        if set(result) != expected_set:
            raise RuntimeError(
                f"Replication {rep_idx} has inconsistent method keys."
            )
        for method, value in result.items():
            if not np.isfinite(value):
                raise RuntimeError(
                    f"Replication {rep_idx}, method '{method}' "
                    f"returned non-finite pseudo-R²: {value}."
                )

    return {
        method: {
            "mean": float(np.mean([r[method] for r in replication_results])),
            "std": (
                float(np.std([r[method] for r in replication_results], ddof=1))
                if len(replication_results) > 1
                else 0.0
            ),
        }
        for method in expected_methods
    }
