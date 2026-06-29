"""
Simulation framework: shared infrastructure plus Study 1 and Study 2 mixins.

Study 1 (Strands 1--3): group-level pseudo-R² — see studies/study1_group_level.py
Study 2 (Strands 4--7): pair-level contrast — see studies/study2_directional_pairs.py
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    from .sequence_generator import SequenceGenerator
    from .distance_computer import DistanceComputer
    from .evaluation import (
        aggregate_replications,
        compute_pseudo_r2_single,
        compute_early_late_directional_scores,
    )
    from .config import SimulationConfig
    from .studies.study1_group_level import GroupLevelStudyMixin
    from .studies.study2_directional_pairs import DirectionalPairStudyMixin
except ImportError:
    from sequence_generator import SequenceGenerator
    from distance_computer import DistanceComputer
    from evaluation import (
        aggregate_replications,
        compute_pseudo_r2_single,
        compute_early_late_directional_scores,
    )
    from config import SimulationConfig
    from studies.study1_group_level import GroupLevelStudyMixin
    from studies.study2_directional_pairs import DirectionalPairStudyMixin


class SimulationFrameworkBase:
    """Shared sequence validation, distance computation, and evaluation helpers."""

    def __init__(self, config: SimulationConfig):
        self.config = config

        self.seq_generator = SequenceGenerator(
            state_labels=config.state_labels,
            total_length=config.total_length,
            random_seed=config.random_seed,
        )

        self.dist_computer = DistanceComputer(
            state_labels=config.state_labels,
            total_length=config.total_length,
            default_norm=config.distance_norm,
        )

        methods = config.get_distance_methods()
        valid_methods, invalid_methods = self.dist_computer.validate_methods(methods)

        if invalid_methods:
            raise ValueError(
                f"Configuration contains unsupported distance methods: {invalid_methods}. "
                f"Valid methods: {sorted(self.dist_computer.allowed_methods)}"
            )

        if self.config.verbose:
            print(
                f"Initialized simulation framework with {len(valid_methods)} distance methods"
            )

    def _assert_total_duration(
        self,
        dss: List[str],
        durations: np.ndarray,
        *,
        context: str = "sequence",
    ) -> None:
        T = self.config.total_length
        if len(dss) != len(durations):
            raise RuntimeError(
                f"{context}: DSS length {len(dss)} != durations length {len(durations)}."
            )
        durations = self.seq_generator._validate_integer_durations(durations)
        invalid_states = [s for s in dss if s not in self.config.state_labels]
        if invalid_states:
            raise RuntimeError(f"{context}: invalid states: {invalid_states}.")
        if int(durations.sum()) != T:
            raise RuntimeError(
                f"{context}: durations sum to {int(durations.sum())}, expected {T}."
            )
        if np.any(durations < 1):
            raise RuntimeError(f"{context}: non-positive spell duration.")
        if any(a == b for a, b in zip(dss, dss[1:])):
            raise RuntimeError(
                f"{context}: adjacent duplicate states in DSS: {dss}."
            )

    def _validate_generated_group(
        self,
        sequences: List[Tuple[List[str], np.ndarray]],
        expected_size: int,
    ) -> None:
        if len(sequences) != expected_size:
            raise RuntimeError(
                f"Expected {expected_size} sequences, got {len(sequences)}."
            )
        for idx, (dss, durations) in enumerate(sequences):
            self._assert_total_duration(dss, durations, context=f"sequence {idx}")

    def _position_to_spells(
        self, position_sequence: List[str]
    ) -> Tuple[List[str], np.ndarray]:
        if len(position_sequence) == 0:
            return ([], np.array([], dtype=int))

        dss = []
        durations = []
        current_state = position_sequence[0]
        current_dur = 1

        for i in range(1, len(position_sequence)):
            if position_sequence[i] == current_state:
                current_dur += 1
            else:
                dss.append(current_state)
                durations.append(current_dur)
                current_state = position_sequence[i]
                current_dur = 1

        dss.append(current_state)
        durations.append(current_dur)

        return (dss, np.array(durations, dtype=int))

    def _compute_distance_matrix_for_method(
        self,
        sequences: List[Tuple[List[str], np.ndarray]],
        method: str,
        strand_name: str,
        replication_id: int,
    ) -> np.ndarray:
        log_calls = getattr(self.config, "log_distance_calls", False)
        common = dict(
            om_indel_cost=self.config.om_indel_cost,
            om_substitution_cost=self.config.om_substitution_cost,
            strand_name=strand_name,
            replication_id=replication_id,
            log_distance=log_calls,
        )
        if (
            method.startswith("LCPspell_expcost_")
            or method.startswith("RLCPspell_expcost_")
            or method.startswith("OMspell_expcost_")
            or method.startswith("OMspellRS_expcost_")
        ):
            expcost = float(method.split("_")[-1])
            base_method = method.split("_expcost_")[0]
            return self.dist_computer.compute_distance_matrix(
                sequences,
                base_method,
                expcost=expcost,
                **common,
            )
        return self.dist_computer.compute_distance_matrix(
            sequences,
            method,
            **common,
        )

    def _evaluate_sequence_groups_for_strand(
        self,
        all_sequences: List[Tuple[List[str], np.ndarray]],
        group_labels: np.ndarray,
        strand_name: str,
        replication_id: int,
    ) -> Dict[str, float]:
        """Study 1 metric: chance-corrected pseudo-R² per method."""
        methods = self.config.get_distance_methods()
        results: Dict[str, float] = {}
        for method in methods:
            dist_matrix = self._compute_distance_matrix_for_method(
                all_sequences,
                method,
                strand_name,
                replication_id,
            )
            results[method] = compute_pseudo_r2_single(dist_matrix, group_labels)
        return results

    def _evaluate_directional_early_late_for_strand(
        self,
        matched_draws: List[
            Tuple[
                Tuple[Tuple[List[str], np.ndarray], Tuple[List[str], np.ndarray]],
                Tuple[Tuple[List[str], np.ndarray], Tuple[List[str], np.ndarray]],
            ]
        ],
        *,
        contrast: str,
        strand_name: str,
        replication_id: int,
    ) -> Dict[str, Dict[str, float]]:
        """
        Study 2 metric: early-vs-late normalized paired contrast per method.

        Each nested matched draw contributes (d_early, d_late) from focal pair
        distances. Distances are computed in chunked batches for efficiency; only
        the two focal within-pair distances from each draw enter the score.
        """
        methods = self.config.get_distance_methods()
        chunk_size = max(1, int(getattr(self.config, "directional_pair_chunk_size", 100)))
        results: Dict[str, Dict[str, float]] = {}

        for method in methods:
            d_early_all: List[float] = []
            d_late_all: List[float] = []

            for chunk_start in range(0, len(matched_draws), chunk_size):
                chunk = matched_draws[chunk_start : chunk_start + chunk_size]
                sequences: List[Tuple[List[str], np.ndarray]] = []
                pair_indices: List[Tuple[int, int, int, int]] = []

                for (early_a, early_b), (late_a, late_b) in chunk:
                    base = len(sequences)
                    sequences.extend([early_a, early_b, late_a, late_b])
                    pair_indices.append((base, base + 1, base + 2, base + 3))

                dist_matrix = self._compute_distance_matrix_for_method(
                    sequences,
                    method,
                    strand_name,
                    replication_id,
                )

                for early_i, early_j, late_i, late_j in pair_indices:
                    d_early_all.append(float(dist_matrix[early_i, early_j]))
                    d_late_all.append(float(dist_matrix[late_i, late_j]))

            d_early = np.asarray(d_early_all, dtype=np.float64)
            d_late = np.asarray(d_late_all, dtype=np.float64)

            contrast_score, win_score = compute_early_late_directional_scores(
                d_early,
                d_late,
                contrast,
            )

            results[method] = {"contrast": contrast_score, "win": win_score}

        return results


class SimulationFramework(
    SimulationFrameworkBase,
    GroupLevelStudyMixin,
    DirectionalPairStudyMixin,
):
    """Full paper simulation engine (Study 1 + Study 2)."""
