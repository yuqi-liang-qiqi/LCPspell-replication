"""
Optional robustness check: localized perturbation location sensitivity.

Not used in main-text Strands 4--7. Answers whether measures respond to where
a fixed-length local deviation occurs, not to persistent divergence/convergence.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

try:
    from ..evaluation import aggregate_replications
except ImportError:
    from evaluation import aggregate_replications

if TYPE_CHECKING:
    pass

SpellSeq = Tuple[List[str], np.ndarray]

# Calendar-time perturbation defaults (T=20, L=3 in the paper).
CALENDAR_WINDOW_LENGTH = 3
CALENDAR_NEAR_ORIGIN_RANGE = (2, 6)


class LocalizedPerturbationMixin:
    """Robustness-only localized perturbation operators and runners."""

    @staticmethod
    def _mirror_calendar_start_range(
        near_origin_range: Tuple[int, int],
        *,
        total_length: int,
        window_length: int,
    ) -> Tuple[int, int]:
        """Return the exact reflected start-position range along a shared time axis."""
        lo, hi = near_origin_range
        if lo < 0 or hi < lo:
            raise ValueError(f"Invalid near_origin_range: {near_origin_range}")

        far_lo = total_length - window_length - hi
        far_hi = total_length - window_length - lo

        if far_lo < 0 or far_hi + window_length > total_length:
            raise ValueError(
                f"Mirrored range {(far_lo, far_hi)} is invalid for "
                f"T={total_length}, L={window_length}."
            )

        return far_lo, far_hi

    def _calendar_ranges(
        self,
        window_length: int = CALENDAR_WINDOW_LENGTH,
        near_origin_range: Tuple[int, int] = CALENDAR_NEAR_ORIGIN_RANGE,
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Near-origin and far-origin calendar start ranges (exact mirror pair)."""
        far_origin = self._mirror_calendar_start_range(
            near_origin_range,
            total_length=self.config.total_length,
            window_length=window_length,
        )
        return near_origin_range, far_origin

    def _reverse_spell_sequence(self, sequence: SpellSeq) -> SpellSeq:
        """Reverse a spell sequence while preserving state-duration pairs."""
        dss, durations = sequence
        out_dss = list(reversed(dss))
        out_durations = self.seq_generator._validate_integer_durations(durations)[::-1].copy()
        self._assert_total_duration(out_dss, out_durations, context="reversed sequence")
        return out_dss, out_durations

    def _sample_random_spell_sequence(self, n_spells: int) -> SpellSeq:
        """Generate one valid random spell sequence with fixed total length."""
        dss = self.seq_generator.generate_random_dss(
            n_spells,
            no_adjacent_repeats=True,
        )
        durations = self.seq_generator.generate_random_durations(n_spells)
        self._assert_total_duration(dss, durations, context="random base sequence")
        return dss, durations

    def _relabel_position_window(
        self,
        sequence: SpellSeq,
        start: int,
        window_length: int,
    ) -> SpellSeq:
        """
        Apply a fixed-length position-level perturbation.

        Every position in [start, start + window_length) is relabelled using a
        deterministic cyclic map. The number of altered calendar positions is
        therefore exactly window_length for every trajectory.
        """
        if window_length < 1:
            raise ValueError("window_length must be >= 1")

        T = self.config.total_length
        end = start + window_length
        if start < 0 or end > T:
            raise ValueError(
                f"Window [{start}, {end}) exceeds total sequence length T={T}."
            )

        labels = self.config.state_labels
        if len(labels) < 2:
            raise ValueError("At least two states are required for relabelling.")

        next_state = {
            labels[i]: labels[(i + 1) % len(labels)]
            for i in range(len(labels))
        }

        dss, durations = sequence
        positions = self.seq_generator.expand_to_position_sequence(dss, durations)
        perturbed_positions = positions.copy()

        for t in range(start, end):
            perturbed_positions[t] = next_state[positions[t]]

        if sum(a != b for a, b in zip(positions, perturbed_positions)) != window_length:
            raise RuntimeError(
                "Calendar perturbation did not alter exactly window_length positions."
            )

        out_dss, out_durations = self._position_to_spells(perturbed_positions)
        self._assert_total_duration(
            out_dss,
            out_durations,
            context="position-window perturbation",
        )
        return out_dss, out_durations

    def _apply_oriented_calendar_perturbation(
        self,
        sequence: SpellSeq,
        start_from_directional_origin: int,
        window_length: int,
        *,
        reverse_direction: bool,
    ) -> SpellSeq:
        """
        Apply a calendar-time perturbation from the relevant directional origin.

        reverse_direction=False:
            count from the beginning of the original trajectory (divergence).

        reverse_direction=True:
            reverse the trajectory, apply the same operator, and reverse it back
            (convergence).
        """
        working_sequence = (
            self._reverse_spell_sequence(sequence)
            if reverse_direction
            else sequence
        )

        perturbed = self._relabel_position_window(
            working_sequence,
            start=start_from_directional_origin,
            window_length=window_length,
        )

        return (
            self._reverse_spell_sequence(perturbed)
            if reverse_direction
            else perturbed
        )

    def _swap_adjacent_spell_pairs(
        self,
        sequence: SpellSeq,
        spell_idx: int,
    ) -> SpellSeq:
        """
        Swap two adjacent complete spells, including both states and durations.

        This preserves the multiset of (state, duration) spell pairs and changes
        only their order.
        """
        dss, durations = sequence
        if spell_idx < 0 or spell_idx >= len(dss) - 1:
            raise ValueError(
                f"spell_idx={spell_idx} out of range for {len(dss)} spells."
            )

        out_dss = list(dss)
        out_durations = self.seq_generator._validate_integer_durations(durations).copy()

        out_dss[spell_idx], out_dss[spell_idx + 1] = (
            out_dss[spell_idx + 1],
            out_dss[spell_idx],
        )
        out_durations[spell_idx], out_durations[spell_idx + 1] = (
            out_durations[spell_idx + 1],
            out_durations[spell_idx],
        )

        if any(a == b for a, b in zip(out_dss, out_dss[1:])):
            raise ValueError(
                f"Adjacent spell swap created duplicate DSS states: {out_dss}"
            )

        self._assert_total_duration(
            out_dss,
            out_durations,
            context="adjacent spell-pair swap",
        )
        return out_dss, out_durations

    def _apply_oriented_spell_order_perturbation(
        self,
        sequence: SpellSeq,
        spell_idx_from_directional_origin: int,
        *,
        reverse_direction: bool,
    ) -> SpellSeq:
        """
        Apply an adjacent spell-pair swap from the relevant directional origin.

        reverse_direction=False:
            count from the sequence beginning (divergence).

        reverse_direction=True:
            reverse the spell sequence, apply the same swap, and reverse it back
            (convergence).
        """
        working_sequence = (
            self._reverse_spell_sequence(sequence)
            if reverse_direction
            else sequence
        )

        perturbed = self._swap_adjacent_spell_pairs(
            working_sequence,
            spell_idx=spell_idx_from_directional_origin,
        )

        return (
            self._reverse_spell_sequence(perturbed)
            if reverse_direction
            else perturbed
        )

    def _sample_spell_sequence_swappable_at(
        self,
        n_spells: int,
        required_indices: List[int],
        *,
        reverse_direction: bool,
        max_attempts: int = 1000,
    ) -> SpellSeq:
        """Sample a base sequence for which all required swaps are valid."""
        for _ in range(max_attempts):
            sequence = self._sample_random_spell_sequence(n_spells)
            try:
                for spell_idx in required_indices:
                    self._apply_oriented_spell_order_perturbation(
                        sequence,
                        spell_idx_from_directional_origin=spell_idx,
                        reverse_direction=reverse_direction,
                    )
                return sequence
            except ValueError:
                continue

        raise RuntimeError(
            "Unable to sample a DSS sequence valid for all requested spell-order swaps."
        )

    def _run_calendar_location_comparison(
        self,
        *,
        strand_name: str,
        reverse_direction: bool,
        group1_range: Tuple[int, int],
        group2_range: Tuple[int, int],
        n_replications: Optional[int] = None,
        n_spells_base: int = 4,
        window_length: int = CALENDAR_WINDOW_LENGTH,
    ) -> Dict[str, Dict[str, float]]:
        """
        Combined calendar-time comparison between two perturbation-location ranges.

        Public wrappers assign group1/group2 ranges to substantive earlier/later
        labels; convergence wrappers swap the ranges relative to divergence.
        """
        if n_replications is None:
            n_replications = self.config.n_replications

        replication_results = []
        n_per_group = self.config.n_sequences_per_group

        for rep in range(n_replications):
            self.seq_generator.rng = np.random.default_rng(
                self.config.random_seed + rep
            )

            group1_sequences = []
            group2_sequences = []

            for _ in range(n_per_group):
                base = self._sample_random_spell_sequence(n_spells_base)

                start1 = int(
                    self.seq_generator.rng.integers(
                        group1_range[0],
                        group1_range[1] + 1,
                    )
                )
                start2 = int(
                    self.seq_generator.rng.integers(
                        group2_range[0],
                        group2_range[1] + 1,
                    )
                )

                group1_sequences.append(
                    self._apply_oriented_calendar_perturbation(
                        base,
                        start_from_directional_origin=start1,
                        window_length=window_length,
                        reverse_direction=reverse_direction,
                    )
                )
                group2_sequences.append(
                    self._apply_oriented_calendar_perturbation(
                        base,
                        start_from_directional_origin=start2,
                        window_length=window_length,
                        reverse_direction=reverse_direction,
                    )
                )

            self._validate_generated_group(group1_sequences, n_per_group)
            self._validate_generated_group(group2_sequences, n_per_group)

            all_sequences = group1_sequences + group2_sequences
            group_labels = np.array([0] * n_per_group + [1] * n_per_group)

            replication_results.append(
                self._evaluate_sequence_groups_for_strand(
                    all_sequences,
                    group_labels,
                    strand_name,
                    rep,
                )
            )

        return aggregate_replications(replication_results)

    def _run_calendar_signal_vs_control(
        self,
        *,
        strand_name: str,
        reverse_direction: bool,
        location_range: Tuple[int, int],
        n_replications: Optional[int] = None,
        n_spells_base: int = 4,
        window_length: int = 3,
    ) -> Dict[str, Dict[str, float]]:
        """Decomposed calendar-time comparison: matched perturbation versus paired control."""
        if n_replications is None:
            n_replications = self.config.n_replications

        replication_results = []
        n_per_group = self.config.n_sequences_per_group

        for rep in range(n_replications):
            self.seq_generator.rng = np.random.default_rng(
                self.config.random_seed + rep
            )

            group_signal = []
            group_control = []

            for _ in range(n_per_group):
                base = self._sample_random_spell_sequence(n_spells_base)
                start = int(
                    self.seq_generator.rng.integers(
                        location_range[0],
                        location_range[1] + 1,
                    )
                )

                group_signal.append(
                    self._apply_oriented_calendar_perturbation(
                        base,
                        start_from_directional_origin=start,
                        window_length=window_length,
                        reverse_direction=reverse_direction,
                    )
                )
                group_control.append((list(base[0]), base[1].copy()))

            self._validate_generated_group(group_signal, n_per_group)
            self._validate_generated_group(group_control, n_per_group)

            all_sequences = group_signal + group_control
            group_labels = np.array([0] * n_per_group + [1] * n_per_group)

            replication_results.append(
                self._evaluate_sequence_groups_for_strand(
                    all_sequences,
                    group_labels,
                    strand_name,
                    rep,
                )
            )

        return aggregate_replications(replication_results)

    def _run_spell_order_location_comparison(
        self,
        *,
        strand_name: str,
        reverse_direction: bool,
        group1_idx: int,
        group2_idx: int,
        n_replications: Optional[int] = None,
        n_spells: int = 5,
    ) -> Dict[str, Dict[str, float]]:
        """Combined spell-order comparison between two adjacent spell-pair swap indices."""
        if n_replications is None:
            n_replications = self.config.n_replications

        replication_results = []
        n_per_group = self.config.n_sequences_per_group

        for rep in range(n_replications):
            self.seq_generator.rng = np.random.default_rng(
                self.config.random_seed + rep
            )

            group1_sequences = []
            group2_sequences = []

            for _ in range(n_per_group):
                base = self._sample_spell_sequence_swappable_at(
                    n_spells,
                    required_indices=[group1_idx, group2_idx],
                    reverse_direction=reverse_direction,
                )

                group1_sequences.append(
                    self._apply_oriented_spell_order_perturbation(
                        base,
                        spell_idx_from_directional_origin=group1_idx,
                        reverse_direction=reverse_direction,
                    )
                )
                group2_sequences.append(
                    self._apply_oriented_spell_order_perturbation(
                        base,
                        spell_idx_from_directional_origin=group2_idx,
                        reverse_direction=reverse_direction,
                    )
                )

            self._validate_generated_group(group1_sequences, n_per_group)
            self._validate_generated_group(group2_sequences, n_per_group)

            all_sequences = group1_sequences + group2_sequences
            group_labels = np.array([0] * n_per_group + [1] * n_per_group)

            replication_results.append(
                self._evaluate_sequence_groups_for_strand(
                    all_sequences,
                    group_labels,
                    strand_name,
                    rep,
                )
            )

        return aggregate_replications(replication_results)

    def _run_spell_order_signal_vs_control(
        self,
        *,
        strand_name: str,
        reverse_direction: bool,
        spell_idx_from_directional_origin: int,
        n_replications: Optional[int] = None,
        n_spells: int = 5,
    ) -> Dict[str, Dict[str, float]]:
        """Decomposed spell-order comparison: adjacent spell-pair swap versus paired control."""
        if n_replications is None:
            n_replications = self.config.n_replications

        replication_results = []
        n_per_group = self.config.n_sequences_per_group

        for rep in range(n_replications):
            self.seq_generator.rng = np.random.default_rng(
                self.config.random_seed + rep
            )

            group_signal = []
            group_control = []

            swap_end_idx = n_spells - 2
            for _ in range(n_per_group):
                base = self._sample_spell_sequence_swappable_at(
                    n_spells,
                    required_indices=[0, swap_end_idx],
                    reverse_direction=reverse_direction,
                )

                group_signal.append(
                    self._apply_oriented_spell_order_perturbation(
                        base,
                        spell_idx_from_directional_origin=spell_idx_from_directional_origin,
                        reverse_direction=reverse_direction,
                    )
                )
                group_control.append((list(base[0]), base[1].copy()))

            self._validate_generated_group(group_signal, n_per_group)
            self._validate_generated_group(group_control, n_per_group)

            all_sequences = group_signal + group_control
            group_labels = np.array([0] * n_per_group + [1] * n_per_group)

            replication_results.append(
                self._evaluate_sequence_groups_for_strand(
                    all_sequences,
                    group_labels,
                    strand_name,
                    rep,
                )
            )

        return aggregate_replications(replication_results)

    # --- Public robustness runners (legacy localized-perturbation keys) ---

    def run_localized_calendar_early_vs_late(
        self,
        n_replications: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Earlier vs later localized calendar-window perturbation (divergence operator)."""
        near_origin, far_origin = self._calendar_ranges()
        return self._run_calendar_location_comparison(
            strand_name="localized_calendar_early_vs_late",
            reverse_direction=False,
            group1_range=near_origin,
            group2_range=far_origin,
            n_replications=n_replications,
        )

    def run_localized_spell_order_early_vs_late(
        self,
        n_replications: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Earlier vs later localized adjacent spell-pair swap (divergence operator)."""
        return self._run_spell_order_location_comparison(
            strand_name="localized_spell_order_early_vs_late",
            reverse_direction=False,
            group1_idx=0,
            group2_idx=3,
            n_replications=n_replications,
        )
