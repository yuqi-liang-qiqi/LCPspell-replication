"""
Study 2 (Strands 4--7): nested matched early-vs-late directional contrasts.

Each draw samples a quadruple ``((early_a, early_b), (late_a, late_b))`` from a
shared latent background so that early and late focal pairs differ only in where
the shared segment occurs. The estimand is whether a distance method responds
more strongly when the directional pattern appears earlier (divergence) or
produces smaller pair distances when convergence is earlier (convergence).

Metric: normalized aggregate early-vs-late contrast (mean focal distances).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Literal, Optional, Tuple

import numpy as np

try:
    from ..evaluation import aggregate_directional_replications
except ImportError:
    from evaluation import aggregate_directional_replications

SpellSeq = Tuple[List[str], np.ndarray]
Pair = Tuple[SpellSeq, SpellSeq]
MatchedDraw = Tuple[Pair, Pair]
DurationMode = Literal[
    "matched", "shared_spell_mismatch_compensated", "background_noise"
]
ContrastMode = Literal["divergence", "convergence"]

# Segment lengths (not Python indices). Onset positions follow the paper mapping.
EARLY_DIVERGENCE_PREFIX_RANGE = (3, 6)
LATE_DIVERGENCE_PREFIX_RANGE = (14, 17)
EARLY_CONVERGENCE_SUFFIX_RANGE = (15, 17)
LATE_CONVERGENCE_SUFFIX_RANGE = (3, 5)

SPELL_COUNT = 5
EARLY_DIVERGENCE_SHARED_PREFIX_SPELLS = 1
LATE_DIVERGENCE_SHARED_PREFIX_SPELLS = 3
EARLY_CONVERGENCE_SHARED_SUFFIX_SPELLS = 4
LATE_CONVERGENCE_SHARED_SUFFIX_SPELLS = 2


class DirectionalPairStudyMixin:
    """Study 2 — matched early-vs-late directional pair contrasts."""

    @staticmethod
    def build_calendar_divergence_templates(
        shared_prefix_positions: List[str],
        group1_suffix_positions: List[str],
        group2_suffix_positions: List[str],
    ) -> Tuple[List[str], List[str]]:
        """Same prefix, persistently different suffixes."""
        if len(group1_suffix_positions) != len(group2_suffix_positions):
            raise ValueError("Suffixes must have equal lengths.")
        if any(a == b for a, b in zip(group1_suffix_positions, group2_suffix_positions)):
            raise ValueError("Divergent suffixes must differ at every position.")
        return (
            shared_prefix_positions + group1_suffix_positions,
            shared_prefix_positions + group2_suffix_positions,
        )

    @staticmethod
    def build_calendar_convergence_templates(
        group1_prefix_positions: List[str],
        group2_prefix_positions: List[str],
        shared_suffix_positions: List[str],
    ) -> Tuple[List[str], List[str]]:
        """Persistently different prefixes, same suffix."""
        if len(group1_prefix_positions) != len(group2_prefix_positions):
            raise ValueError("Prefixes must have equal lengths.")
        if any(a == b for a, b in zip(group1_prefix_positions, group2_prefix_positions)):
            raise ValueError("Pre-convergence prefixes must differ at every position.")
        return (
            group1_prefix_positions + shared_suffix_positions,
            group2_prefix_positions + shared_suffix_positions,
        )

    @staticmethod
    def build_spell_order_divergence_templates(
        shared_prefix: List[str],
        suffix1: List[str],
        suffix2: List[str],
    ) -> Tuple[List[str], List[str]]:
        """Same DSS prefix, persistently different DSS suffixes."""
        if len(suffix1) != len(suffix2):
            raise ValueError("DSS suffixes must have equal lengths.")
        if any(a == b for a, b in zip(suffix1, suffix2)):
            raise ValueError("Divergent DSS suffixes must differ elementwise.")
        return shared_prefix + suffix1, shared_prefix + suffix2

    @staticmethod
    def build_spell_order_convergence_templates(
        prefix1: List[str],
        prefix2: List[str],
        shared_suffix: List[str],
    ) -> Tuple[List[str], List[str]]:
        """Different DSS prefixes, same DSS suffix."""
        if len(prefix1) != len(prefix2):
            raise ValueError("DSS prefixes must have equal lengths.")
        if any(a == b for a, b in zip(prefix1, prefix2)):
            raise ValueError("Pre-convergence DSS prefixes must differ elementwise.")
        return prefix1 + shared_suffix, prefix2 + shared_suffix

    def _directional_duration_mode(self) -> DurationMode:
        mode = getattr(self.config, "directional_duration_mode", "matched")
        if mode == "shared_spell_mismatch":
            mode = "shared_spell_mismatch_compensated"
        allowed = {
            "matched",
            "shared_spell_mismatch_compensated",
            "background_noise",
        }
        if mode not in allowed:
            raise ValueError(
                f"directional_duration_mode must be one of {sorted(allowed)}, got {mode!r}."
            )
        return mode  # type: ignore[return-value]

    def _random_position_segment(self, length: int) -> List[str]:
        if length < 1:
            raise ValueError("Segment length must be >= 1.")
        return list(
            self.seq_generator.rng.choice(self.config.state_labels, size=length)
        )

    def _sample_position_segment_differing_from(
        self,
        reference: List[str],
        *,
        max_attempts: int = 500,
    ) -> List[str]:
        for _ in range(max_attempts):
            seg = self._random_position_segment(len(reference))
            if all(a != b for a, b in zip(seg, reference)):
                return seg
        raise RuntimeError(
            f"Unable to sample a position segment differing from reference "
            f"at every position (length {len(reference)})."
        )

    def _sample_differing_position_segments(
        self,
        length: int,
        *,
        max_attempts: int = 500,
    ) -> Tuple[List[str], List[str]]:
        for _ in range(max_attempts):
            seg1 = self._random_position_segment(length)
            seg2 = self._random_position_segment(length)
            if all(a != b for a, b in zip(seg1, seg2)):
                return seg1, seg2
        raise RuntimeError(
            f"Unable to sample elementwise-different position segments of length {length}."
        )

    def _sample_dss_segment_differing_from(
        self,
        reference: List[str],
        *,
        left_boundary: Optional[str] = None,
        right_boundary: Optional[str] = None,
        max_attempts: int = 1000,
    ) -> List[str]:
        for _ in range(max_attempts):
            seg = self.seq_generator.generate_random_dss(
                len(reference), no_adjacent_repeats=True
            )
            if not all(a != b for a, b in zip(seg, reference)):
                continue
            if left_boundary is not None and seg[0] == left_boundary:
                continue
            if right_boundary is not None and seg[-1] == right_boundary:
                continue
            return seg
        raise RuntimeError(
            f"Unable to sample DSS segment differing from reference "
            f"at every position (length {len(reference)})."
        )

    def _sample_differing_dss_segments(
        self,
        length: int,
        *,
        left_boundary: Optional[str] = None,
        right_boundary: Optional[str] = None,
        max_attempts: int = 1000,
    ) -> Tuple[List[str], List[str]]:
        for _ in range(max_attempts):
            seg1 = self.seq_generator.generate_random_dss(
                length, no_adjacent_repeats=True
            )
            seg2 = self.seq_generator.generate_random_dss(
                length, no_adjacent_repeats=True
            )
            if not all(a != b for a, b in zip(seg1, seg2)):
                continue
            if left_boundary is not None:
                if seg1[0] == left_boundary or seg2[0] == left_boundary:
                    continue
            if right_boundary is not None:
                if seg1[-1] == right_boundary or seg2[-1] == right_boundary:
                    continue
            return seg1, seg2
        raise RuntimeError(
            f"Unable to sample elementwise-different DSS segments of length {length}."
        )

    @staticmethod
    def _has_adjacent_duplicates(dss: List[str]) -> bool:
        return any(a == b for a, b in zip(dss, dss[1:]))

    def _durations_matched(self, n_spells: int) -> np.ndarray:
        return self.seq_generator.generate_random_durations(n_spells)

    def _durations_background_noise(self, n_spells: int) -> np.ndarray:
        return self.seq_generator.generate_random_durations(n_spells)

    def _durations_shared_spell_mismatch_compensated(
        self,
        n_spells: int,
        shared_indices: List[int],
        *,
        max_attempts: int = 500,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Introduce a one-unit duration mismatch on one shared spell, compensated
        by a one-unit transfer from another spell so total length stays fixed.
        """
        if not shared_indices:
            dur = self._durations_matched(n_spells)
            return dur.copy(), dur.copy()

        for _ in range(max_attempts):
            dur_a = self._durations_matched(n_spells)
            dur_b = dur_a.copy()
            idx = int(self.seq_generator.rng.choice(shared_indices))
            donor_candidates = [i for i in range(n_spells) if i != idx and dur_b[i] > 1]
            if not donor_candidates:
                continue
            donor = int(self.seq_generator.rng.choice(donor_candidates))
            dur_b[idx] += 1
            dur_b[donor] -= 1
            if dur_b[idx] >= 1 and dur_b[donor] >= 1:
                return dur_a, dur_b

        raise RuntimeError(
            "Unable to sample compensated shared-spell duration mismatch "
            f"for indices {shared_indices}."
        )

    def _spell_durations_for_quadruple(
        self,
        *,
        n_spells: int,
        shared_indices_early: List[int],
        shared_indices_late: List[int],
        duration_mode: DurationMode,
        dss_early_a: List[str],
        dss_early_b: List[str],
        dss_late_a: List[str],
        dss_late_b: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Spell-order durations for one nested early/late matched draw."""
        if duration_mode == "matched":
            dur = self._durations_matched(n_spells)
            return dur.copy(), dur.copy(), dur.copy(), dur.copy()

        if duration_mode == "background_noise":
            return (
                self._durations_background_noise(n_spells),
                self._durations_background_noise(n_spells),
                self._durations_background_noise(n_spells),
                self._durations_background_noise(n_spells),
            )

        if duration_mode == "shared_spell_mismatch_compensated":
            dur_ea, dur_eb = self._spell_durations_for_pair(
                dss_early_a,
                dss_early_b,
                shared_indices=shared_indices_early,
                duration_mode=duration_mode,
            )
            dur_la, dur_lb = self._spell_durations_for_pair(
                dss_late_a,
                dss_late_b,
                shared_indices=shared_indices_late,
                duration_mode=duration_mode,
            )
            return dur_ea, dur_eb, dur_la, dur_lb

        raise ValueError(f"Unknown duration_mode: {duration_mode!r}.")

    def _spell_durations_for_pair(
        self,
        dss_a: List[str],
        dss_b: List[str],
        *,
        shared_indices: List[int],
        duration_mode: DurationMode,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_spells = len(dss_a)
        if len(dss_b) != n_spells:
            raise ValueError("Pair trajectories must have the same spell count.")

        if duration_mode == "matched":
            dur = self._durations_matched(n_spells)
            return dur.copy(), dur.copy()

        if duration_mode == "background_noise":
            return (
                self._durations_background_noise(n_spells),
                self._durations_background_noise(n_spells),
            )

        if duration_mode == "shared_spell_mismatch_compensated":
            return self._durations_shared_spell_mismatch_compensated(
                n_spells, shared_indices
            )

        raise ValueError(f"Unknown duration_mode: {duration_mode!r}.")

    def _sample_calendar_divergence_pair(
        self,
        prefix_len_range: Tuple[int, int],
    ) -> Pair:
        """Shared prefix, elementwise-different suffixes."""
        prefix_len = int(
            self.seq_generator.rng.integers(
                prefix_len_range[0], prefix_len_range[1] + 1
            )
        )
        suffix_len = self.config.total_length - prefix_len
        shared_prefix = self._random_position_segment(prefix_len)
        suffix_a, suffix_b = self._sample_differing_position_segments(suffix_len)
        pos_a, pos_b = self.build_calendar_divergence_templates(
            shared_prefix, suffix_a, suffix_b
        )
        return (
            self._position_to_spells(pos_a),
            self._position_to_spells(pos_b),
        )

    def _sample_calendar_convergence_pair(
        self,
        suffix_len_range: Tuple[int, int],
    ) -> Pair:
        """Different prefixes, shared suffix."""
        suffix_len = int(
            self.seq_generator.rng.integers(
                suffix_len_range[0], suffix_len_range[1] + 1
            )
        )
        prefix_len = self.config.total_length - suffix_len
        prefix1, prefix2 = self._sample_differing_position_segments(prefix_len)
        shared_suffix = self._random_position_segment(suffix_len)
        pos_a, pos_b = self.build_calendar_convergence_templates(
            prefix1, prefix2, shared_suffix
        )
        return (
            self._position_to_spells(pos_a),
            self._position_to_spells(pos_b),
        )

    def _sample_spell_order_divergence_pair(
        self,
        shared_prefix_len: int,
        *,
        n_spells: int = SPELL_COUNT,
    ) -> Pair:
        suffix_len = n_spells - shared_prefix_len
        duration_mode = self._directional_duration_mode()
        shared_indices = list(range(shared_prefix_len))

        for _ in range(1000):
            shared_prefix = self.seq_generator.generate_random_dss(
                shared_prefix_len, no_adjacent_repeats=True
            )
            suffix_a, suffix_b = self._sample_differing_dss_segments(
                suffix_len,
                left_boundary=shared_prefix[-1] if shared_prefix else None,
            )
            dss_a, dss_b = self.build_spell_order_divergence_templates(
                shared_prefix, suffix_a, suffix_b
            )
            if self._has_adjacent_duplicates(dss_a):
                continue
            if self._has_adjacent_duplicates(dss_b):
                continue

            dur_a, dur_b = self._spell_durations_for_pair(
                dss_a,
                dss_b,
                shared_indices=shared_indices,
                duration_mode=duration_mode,
            )
            return (dss_a, dur_a), (dss_b, dur_b)

        raise RuntimeError(
            "Unable to sample spell-order divergence pair with valid DSS."
        )

    def _sample_spell_order_convergence_pair(
        self,
        shared_suffix_len: int,
        *,
        n_spells: int = SPELL_COUNT,
    ) -> Pair:
        prefix_len = n_spells - shared_suffix_len
        duration_mode = self._directional_duration_mode()
        shared_indices = list(range(prefix_len, n_spells))

        for _ in range(1000):
            shared_suffix = self.seq_generator.generate_random_dss(
                shared_suffix_len, no_adjacent_repeats=True
            )
            prefix1, prefix2 = self._sample_differing_dss_segments(
                prefix_len,
                right_boundary=shared_suffix[0] if shared_suffix else None,
            )
            dss_a, dss_b = self.build_spell_order_convergence_templates(
                prefix1, prefix2, shared_suffix
            )
            if self._has_adjacent_duplicates(dss_a):
                continue
            if self._has_adjacent_duplicates(dss_b):
                continue

            dur_a, dur_b = self._spell_durations_for_pair(
                dss_a,
                dss_b,
                shared_indices=shared_indices,
                duration_mode=duration_mode,
            )
            return (dss_a, dur_a), (dss_b, dur_b)

        raise RuntimeError(
            "Unable to sample spell-order convergence pair with valid DSS."
        )

    def _sample_nested_calendar_divergence_draw(
        self,
        early_prefix_range: Tuple[int, int],
        late_prefix_range: Tuple[int, int],
    ) -> MatchedDraw:
        """
        Nested matched quadruple: early and late focal pairs share the same
        latent prefix/tail states; only the divergence onset differs.
        """
        T = self.config.total_length
        L_early = int(
            self.seq_generator.rng.integers(
                early_prefix_range[0], early_prefix_range[1] + 1
            )
        )
        L_late = int(
            self.seq_generator.rng.integers(
                late_prefix_range[0], late_prefix_range[1] + 1
            )
        )
        if L_early >= L_late:
            raise RuntimeError(
                f"Calendar divergence requires L_early < L_late, got {L_early} and {L_late}."
            )

        shared_early = self._random_position_segment(L_early)
        middle_len = L_late - L_early
        middle_shared = self._random_position_segment(middle_len)
        divergent_middle_a, divergent_middle_b = self._sample_differing_position_segments(
            middle_len
        )
        tail_len = T - L_late
        suffix_a, suffix_b = self._sample_differing_position_segments(tail_len)

        shared_late = shared_early + middle_shared
        late_pos_a = shared_late + suffix_a
        late_pos_b = shared_late + suffix_b
        early_pos_a = shared_early + divergent_middle_a + suffix_a
        early_pos_b = shared_early + divergent_middle_b + suffix_b

        early_pair = (
            self._position_to_spells(early_pos_a),
            self._position_to_spells(early_pos_b),
        )
        late_pair = (
            self._position_to_spells(late_pos_a),
            self._position_to_spells(late_pos_b),
        )
        return early_pair, late_pair

    def _sample_nested_calendar_convergence_draw(
        self,
        early_suffix_range: Tuple[int, int],
        late_suffix_range: Tuple[int, int],
    ) -> MatchedDraw:
        """Nested matched quadruple for calendar-time convergence."""
        T = self.config.total_length
        kappa_early = int(
            self.seq_generator.rng.integers(
                early_suffix_range[0], early_suffix_range[1] + 1
            )
        )
        kappa_late = int(
            self.seq_generator.rng.integers(
                late_suffix_range[0], late_suffix_range[1] + 1
            )
        )
        if kappa_early <= kappa_late:
            raise RuntimeError(
                f"Calendar convergence requires kappa_early > kappa_late, "
                f"got {kappa_early} and {kappa_late}."
            )

        shared_suffix_early = self._random_position_segment(kappa_early)
        shared_suffix_late = shared_suffix_early[-kappa_late:]
        middle_len = kappa_early - kappa_late

        early_prefix_len = T - kappa_early
        early_prefix_a, early_prefix_b = self._sample_differing_position_segments(
            early_prefix_len
        )
        late_divergent_middle_a, late_divergent_middle_b = (
            self._sample_differing_position_segments(middle_len)
        )

        late_prefix_a = early_prefix_a + late_divergent_middle_a
        late_prefix_b = early_prefix_b + late_divergent_middle_b
        early_pos_a = early_prefix_a + shared_suffix_early
        early_pos_b = early_prefix_b + shared_suffix_early
        late_pos_a = late_prefix_a + shared_suffix_late
        late_pos_b = late_prefix_b + shared_suffix_late

        early_pair = (
            self._position_to_spells(early_pos_a),
            self._position_to_spells(early_pos_b),
        )
        late_pair = (
            self._position_to_spells(late_pos_a),
            self._position_to_spells(late_pos_b),
        )
        return early_pair, late_pair

    def _sample_nested_spell_order_divergence_draw(
        self,
        early_shared_prefix_len: int,
        late_shared_prefix_len: int,
        *,
        n_spells: int = SPELL_COUNT,
    ) -> MatchedDraw:
        """Nested spell-order divergence quadruple with shared tail states."""
        if early_shared_prefix_len >= late_shared_prefix_len:
            raise ValueError(
                "Spell-order divergence requires early_shared_prefix_len "
                f"< late_shared_prefix_len, got {early_shared_prefix_len} "
                f"and {late_shared_prefix_len}."
            )

        duration_mode = self._directional_duration_mode()
        middle_len = late_shared_prefix_len - early_shared_prefix_len
        suffix_len = n_spells - late_shared_prefix_len
        shared_indices_late = list(range(late_shared_prefix_len))
        shared_indices_early = list(range(early_shared_prefix_len))

        for _ in range(1000):
            shared_late_prefix = self.seq_generator.generate_random_dss(
                late_shared_prefix_len, no_adjacent_repeats=True
            )
            shared_early_prefix = list(shared_late_prefix[:early_shared_prefix_len])

            suffix_a, suffix_b = self._sample_differing_dss_segments(
                suffix_len,
                left_boundary=shared_late_prefix[-1] if shared_late_prefix else None,
            )
            divergent_middle_a, divergent_middle_b = self._sample_differing_dss_segments(
                middle_len,
                left_boundary=(
                    shared_early_prefix[-1] if shared_early_prefix else None
                ),
                right_boundary=suffix_a[0] if suffix_a else None,
            )

            dss_late_a, dss_late_b = self.build_spell_order_divergence_templates(
                shared_late_prefix, suffix_a, suffix_b
            )
            dss_early_a = shared_early_prefix + divergent_middle_a + suffix_a
            dss_early_b = shared_early_prefix + divergent_middle_b + suffix_b

            if any(
                self._has_adjacent_duplicates(dss)
                for dss in (dss_early_a, dss_early_b, dss_late_a, dss_late_b)
            ):
                continue

            dur_ea, dur_eb, dur_la, dur_lb = self._spell_durations_for_quadruple(
                n_spells=n_spells,
                shared_indices_early=shared_indices_early,
                shared_indices_late=shared_indices_late,
                duration_mode=duration_mode,
                dss_early_a=dss_early_a,
                dss_early_b=dss_early_b,
                dss_late_a=dss_late_a,
                dss_late_b=dss_late_b,
            )
            return (
                (dss_early_a, dur_ea),
                (dss_early_b, dur_eb),
            ), (
                (dss_late_a, dur_la),
                (dss_late_b, dur_lb),
            )

        raise RuntimeError(
            "Unable to sample nested spell-order divergence draw with valid DSS."
        )

    def _sample_nested_spell_order_convergence_draw(
        self,
        early_shared_suffix_len: int,
        late_shared_suffix_len: int,
        *,
        n_spells: int = SPELL_COUNT,
    ) -> MatchedDraw:
        """Nested spell-order convergence quadruple with shared tail states."""
        if early_shared_suffix_len <= late_shared_suffix_len:
            raise ValueError(
                "Spell-order convergence requires early_shared_suffix_len "
                f"> late_shared_suffix_len, got {early_shared_suffix_len} "
                f"and {late_shared_suffix_len}."
            )

        duration_mode = self._directional_duration_mode()
        prefix_len_early = n_spells - early_shared_suffix_len
        middle_len = early_shared_suffix_len - late_shared_suffix_len
        shared_indices_early = list(range(prefix_len_early, n_spells))
        shared_indices_late = list(range(n_spells - late_shared_suffix_len, n_spells))

        for _ in range(1000):
            shared_suffix_early = self.seq_generator.generate_random_dss(
                early_shared_suffix_len, no_adjacent_repeats=True
            )
            shared_suffix_late = list(shared_suffix_early[-late_shared_suffix_len:])

            early_prefix_a, early_prefix_b = self._sample_differing_dss_segments(
                prefix_len_early,
                right_boundary=shared_suffix_early[0],
            )
            late_divergent_middle_a, late_divergent_middle_b = (
                self._sample_differing_dss_segments(
                    middle_len,
                    right_boundary=shared_suffix_late[0],
                )
            )
            if early_prefix_a and late_divergent_middle_a[0] == early_prefix_a[-1]:
                continue
            if early_prefix_b and late_divergent_middle_b[0] == early_prefix_b[-1]:
                continue

            dss_early_a, dss_early_b = self.build_spell_order_convergence_templates(
                early_prefix_a, early_prefix_b, shared_suffix_early
            )
            dss_late_a = early_prefix_a + late_divergent_middle_a + shared_suffix_late
            dss_late_b = early_prefix_b + late_divergent_middle_b + shared_suffix_late

            if any(
                self._has_adjacent_duplicates(dss)
                for dss in (dss_early_a, dss_early_b, dss_late_a, dss_late_b)
            ):
                continue

            dur_ea, dur_eb, dur_la, dur_lb = self._spell_durations_for_quadruple(
                n_spells=n_spells,
                shared_indices_early=shared_indices_early,
                shared_indices_late=shared_indices_late,
                duration_mode=duration_mode,
                dss_early_a=dss_early_a,
                dss_early_b=dss_early_b,
                dss_late_a=dss_late_a,
                dss_late_b=dss_late_b,
            )
            return (
                (dss_early_a, dur_ea),
                (dss_early_b, dur_eb),
            ), (
                (dss_late_a, dur_la),
                (dss_late_b, dur_lb),
            )

        raise RuntimeError(
            "Unable to sample nested spell-order convergence draw with valid DSS."
        )

    def _run_directional_early_late_strand(
        self,
        *,
        strand_name: str,
        contrast: ContrastMode,
        sample_matched_draw: Callable[[], MatchedDraw],
        n_replications: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        if n_replications is None:
            n_replications = self.config.n_replications

        replication_results = []
        n_pairs = self.config.n_sequences_per_group

        for rep in range(n_replications):
            # Intentionally reuse replication seeds across Study~2 strands as a
            # common-random-number design. Each strand resets the RNG independently,
            # so results remain invariant to execution order.
            self.seq_generator.rng = np.random.default_rng(
                self.config.random_seed + rep
            )

            matched_draws: List[MatchedDraw] = []
            for _ in range(n_pairs):
                early, late = sample_matched_draw()
                for label, pair in (("early", early), ("late", late)):
                    for idx, seq in enumerate(pair):
                        self._assert_total_duration(
                            seq[0],
                            seq[1],
                            context=f"{label} pair sequence {idx}",
                        )
                matched_draws.append((early, late))

            replication_results.append(
                self._evaluate_directional_early_late_for_strand(
                    matched_draws,
                    contrast=contrast,
                    strand_name=strand_name,
                    replication_id=rep,
                )
            )

            if self.config.verbose and (rep + 1) % 10 == 0:
                print(
                    f"  [{strand_name}] completed {rep + 1}/{n_replications} replications"
                )

        return aggregate_directional_replications(replication_results)

    # --- Public strand runners (Strands 4--7) ---

    def run_sensitivity_divergence_calendar(
        self,
        n_replications: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        return self._run_directional_early_late_strand(
            strand_name="sensitivity_divergence_calendar",
            contrast="divergence",
            sample_matched_draw=lambda: self._sample_nested_calendar_divergence_draw(
                EARLY_DIVERGENCE_PREFIX_RANGE,
                LATE_DIVERGENCE_PREFIX_RANGE,
            ),
            n_replications=n_replications,
        )

    def run_sensitivity_convergence_calendar(
        self,
        n_replications: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        return self._run_directional_early_late_strand(
            strand_name="sensitivity_convergence_calendar",
            contrast="convergence",
            sample_matched_draw=lambda: self._sample_nested_calendar_convergence_draw(
                EARLY_CONVERGENCE_SUFFIX_RANGE,
                LATE_CONVERGENCE_SUFFIX_RANGE,
            ),
            n_replications=n_replications,
        )

    def run_sensitivity_divergence_spell_order(
        self,
        n_replications: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        return self._run_directional_early_late_strand(
            strand_name="sensitivity_divergence_spell_order",
            contrast="divergence",
            sample_matched_draw=lambda: self._sample_nested_spell_order_divergence_draw(
                EARLY_DIVERGENCE_SHARED_PREFIX_SPELLS,
                LATE_DIVERGENCE_SHARED_PREFIX_SPELLS,
            ),
            n_replications=n_replications,
        )

    def run_sensitivity_convergence_spell_order(
        self,
        n_replications: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        return self._run_directional_early_late_strand(
            strand_name="sensitivity_convergence_spell_order",
            contrast="convergence",
            sample_matched_draw=lambda: self._sample_nested_spell_order_convergence_draw(
                EARLY_CONVERGENCE_SHARED_SUFFIX_SPELLS,
                LATE_CONVERGENCE_SHARED_SUFFIX_SPELLS,
            ),
            n_replications=n_replications,
        )


# Backward-compatible aliases (deprecated).
PersistentDirectionalMixin = DirectionalPairStudyMixin
