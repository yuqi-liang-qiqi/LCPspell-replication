"""
Study 1 (Strands 1--3): group-level sensitivity to timing, sequencing, and duration.

Paper metric: chance-corrected pseudo-R² on balanced two-group designs.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from ..evaluation import aggregate_replications
except ImportError:
    from evaluation import aggregate_replications


class GroupLevelStudyMixin:
    """Study 1 — mixed-perturbation group contrasts (Studer & Ritschard 2016)."""

    def run_sequencing_sensitivity(
        self,
        dss_group1: List[str],
        dss_group2: List[str],
        n_replications: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Strand 2: fixed DSS templates, random durations per sequence."""
        if n_replications is None:
            n_replications = self.config.n_replications

        if len(dss_group1) != len(dss_group2):
            raise ValueError(
                f"DSS templates should have same length for fair comparison: "
                f"{len(dss_group1)} != {len(dss_group2)}"
            )

        replication_results = []

        for rep in range(n_replications):
            self.seq_generator.rng = np.random.default_rng(self.config.random_seed + rep)

            group1_sequences = []
            group2_sequences = []
            for _ in range(self.config.n_sequences_per_group):
                durations1 = self.seq_generator.generate_random_durations(len(dss_group1))
                group1_sequences.append((dss_group1.copy(), durations1))
                durations2 = self.seq_generator.generate_random_durations(len(dss_group2))
                group2_sequences.append((dss_group2.copy(), durations2))

            all_sequences = group1_sequences + group2_sequences
            group_labels = np.array(
                [0] * len(group1_sequences) + [1] * len(group2_sequences)
            )
            replication_results.append(
                self._evaluate_sequence_groups_for_strand(
                    all_sequences, group_labels, "sequencing", rep
                )
            )

            if self.config.verbose and (rep + 1) % 10 == 0:
                print(f"  Completed {rep + 1}/{n_replications} replications")

        return aggregate_replications(replication_results)

    def run_timing_sensitivity(
        self,
        focal_state: str,
        time_group1: int,
        time_group2_range: Tuple[int, int],
        dss_templates: Optional[List[List[str]]] = None,
        n_replications: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Strand 1: focal-state occupancy at earlier vs later calendar positions."""
        if n_replications is None:
            n_replications = self.config.n_replications

        if dss_templates is None:
            dss_templates = [
                self.config.state_labels.copy(),
                list(reversed(self.config.state_labels.copy())),
            ]

        replication_results = []

        for rep in range(n_replications):
            self.seq_generator.rng = np.random.default_rng(self.config.random_seed + rep)

            group1_sequences = []
            group2_sequences = []

            for _ in range(self.config.n_sequences_per_group):
                template = dss_templates[
                    self.seq_generator.rng.integers(len(dss_templates))
                ]
                if focal_state not in template:
                    raise ValueError(
                        f"Every timing DSS template must contain focal_state "
                        f"'{focal_state}'; got {template}"
                    )
                dss1, durations1 = self.seq_generator.sample_sequence_with_focal_state_at_time(
                    template, focal_state, time_group1
                )
                group1_sequences.append((dss1, durations1))

                time_group2 = int(
                    self.seq_generator.rng.integers(
                        time_group2_range[0], time_group2_range[1] + 1
                    )
                )
                template2 = dss_templates[
                    self.seq_generator.rng.integers(len(dss_templates))
                ]
                dss2, durations2 = self.seq_generator.sample_sequence_with_focal_state_at_time(
                    template2, focal_state, time_group2
                )
                group2_sequences.append((dss2, durations2))

            all_sequences = group1_sequences + group2_sequences
            group_labels = np.array(
                [0] * len(group1_sequences) + [1] * len(group2_sequences)
            )
            replication_results.append(
                self._evaluate_sequence_groups_for_strand(
                    all_sequences, group_labels, "timing", rep
                )
            )

            if self.config.verbose and (rep + 1) % 10 == 0:
                print(f"  Completed {rep + 1}/{n_replications} replications")

        return aggregate_replications(replication_results)

    def run_duration_sensitivity(
        self,
        focal_state: str,
        duration_group1: int,
        duration_group2: int,
        dss_templates: Optional[List[List[str]]] = None,
        n_replications: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Strand 3: focal spell duration differs between groups."""
        if n_replications is None:
            n_replications = self.config.n_replications

        if duration_group1 < 1 or duration_group2 < 1:
            raise ValueError("Focal spell durations must be >= 1.")

        if duration_group1 >= self.config.total_length:
            raise ValueError(
                "duration_group1 must leave at least one time unit for other spells."
            )

        if duration_group2 >= self.config.total_length:
            raise ValueError(
                "duration_group2 must leave at least one time unit for other spells."
            )

        if dss_templates is not None:
            for template in dss_templates:
                if template.count(focal_state) != 1:
                    raise ValueError(
                        f"Each duration-strand DSS template must contain focal_state "
                        f"'{focal_state}' exactly once; got {template}."
                    )
                n_other = len(template) - 1
                if self.config.total_length - duration_group1 < n_other:
                    raise ValueError(
                        f"duration_group1={duration_group1} is infeasible for template {template}."
                    )
                if self.config.total_length - duration_group2 < n_other:
                    raise ValueError(
                        f"duration_group2={duration_group2} is infeasible for template {template}."
                    )
        else:
            min_n_other = 2
            if self.config.total_length - duration_group1 < min_n_other:
                raise ValueError(
                    f"duration_group1={duration_group1} is infeasible when DSS templates are "
                    f"random (requires at least {min_n_other} other spells of duration >= 1)."
                )
            if self.config.total_length - duration_group2 < min_n_other:
                raise ValueError(
                    f"duration_group2={duration_group2} is infeasible when DSS templates are "
                    f"random (requires at least {min_n_other} other spells of duration >= 1)."
                )

        replication_results = []

        for rep in range(n_replications):
            self.seq_generator.rng = np.random.default_rng(self.config.random_seed + rep)

            n_per_group = self.config.n_sequences_per_group
            group1_sequences = []
            group2_sequences = []

            while len(group1_sequences) < n_per_group:
                if dss_templates is None:
                    n_spells = self.seq_generator.rng.integers(3, 6)
                    dss_template_this_seq = self.seq_generator.generate_random_dss(
                        n_spells, no_adjacent_repeats=True
                    )
                else:
                    template_idx = self.seq_generator.rng.integers(len(dss_templates))
                    dss_template_this_seq = dss_templates[template_idx].copy()

                if focal_state not in dss_template_this_seq:
                    insert_idx = self.seq_generator.rng.integers(
                        len(dss_template_this_seq) + 1
                    )
                    dss_template_this_seq.insert(insert_idx, focal_state)

                if dss_template_this_seq.count(focal_state) != 1:
                    continue

                focal_spell_index = dss_template_this_seq.index(focal_state)
                remaining1 = self.config.total_length - duration_group1
                if remaining1 < len(dss_template_this_seq) - 1:
                    continue

                other_indices = [
                    i for i in range(len(dss_template_this_seq)) if i != focal_spell_index
                ]
                if other_indices:
                    other_durations1 = self.seq_generator.generate_random_durations(
                        len(other_indices), total_length=remaining1
                    )
                    durations1 = np.zeros(len(dss_template_this_seq), dtype=int)
                    durations1[focal_spell_index] = duration_group1
                    for idx, dur in zip(other_indices, other_durations1):
                        durations1[idx] = dur
                else:
                    durations1 = np.array([duration_group1])

                group1_sequences.append((dss_template_this_seq.copy(), durations1))

            while len(group2_sequences) < n_per_group:
                if dss_templates is None:
                    n_spells2 = self.seq_generator.rng.integers(3, 6)
                    dss_template_this_seq2 = self.seq_generator.generate_random_dss(
                        n_spells2, no_adjacent_repeats=True
                    )
                else:
                    template_idx2 = self.seq_generator.rng.integers(len(dss_templates))
                    dss_template_this_seq2 = dss_templates[template_idx2].copy()

                if focal_state not in dss_template_this_seq2:
                    insert_idx = self.seq_generator.rng.integers(
                        len(dss_template_this_seq2) + 1
                    )
                    dss_template_this_seq2.insert(insert_idx, focal_state)

                if dss_template_this_seq2.count(focal_state) != 1:
                    continue

                focal_spell_index2 = dss_template_this_seq2.index(focal_state)
                remaining2_check = self.config.total_length - duration_group2
                if remaining2_check < len(dss_template_this_seq2) - 1:
                    continue

                other_indices2 = [
                    i for i in range(len(dss_template_this_seq2))
                    if i != focal_spell_index2
                ]
                if other_indices2:
                    other_durations2 = self.seq_generator.generate_random_durations(
                        len(other_indices2), total_length=remaining2_check
                    )
                    durations2 = np.zeros(len(dss_template_this_seq2), dtype=int)
                    durations2[focal_spell_index2] = duration_group2
                    for idx, dur in zip(other_indices2, other_durations2):
                        durations2[idx] = dur
                else:
                    durations2 = np.array([duration_group2], dtype=int)

                group2_sequences.append((dss_template_this_seq2.copy(), durations2))

            self._validate_generated_group(group1_sequences, n_per_group)
            self._validate_generated_group(group2_sequences, n_per_group)

            all_sequences = group1_sequences + group2_sequences
            group_labels = np.array([0] * n_per_group + [1] * n_per_group)

            replication_results.append(
                self._evaluate_sequence_groups_for_strand(
                    all_sequences, group_labels, "duration", rep
                )
            )

            if self.config.verbose and (rep + 1) % 10 == 0:
                print(f"  Completed {rep + 1}/{n_replications} replications")

        return aggregate_replications(replication_results)
