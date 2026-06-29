"""
Supplementary data-generating checks aligned with Studer & Ritschard (2016).

This module implements two families of supplementary checks:
1) Event-based simulations:
   - event-order strand
   - event-timing strand
   - inter-event-duration strand
2) Small perturbation simulations:
   - random token perturbation
   - boundary perturbation (spell-boundary shift)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from ..config import SimulationConfig
from ..distance_computer import DistanceComputer
from ..evaluation import aggregate_replications, compute_pseudo_r2_single
from ..sequence_generator import SequenceGenerator


SpellSeq = Tuple[List[str], np.ndarray]
EventTuple = Tuple[int, int, int]  # e1, e2, e3

STRAND_SEED_OFFSETS: Dict[str, int] = {
    "supplementary_event_order": 1000,
    "supplementary_event_timing": 2000,
    "supplementary_event_inter_duration": 3000,
    "supplementary_small_perturbation_token": 4000,
    "supplementary_small_perturbation_boundary": 5000,
}

# Python indices for event-time ranges (calendar positions are +1).
EVENT_TIMING_EARLY_RANGE = (3, 6)
EVENT_TIMING_LATE_RANGE = (11, 14)
INTER_EVENT_SHORT_GAP = (1, 3)
INTER_EVENT_LONG_GAP = (6, 8)


def supplementary_dg_checks_metadata_extra() -> Dict[str, object]:
    """Single source of truth for supplementary data-generating-check metadata."""
    return {
        "event_timing_early_range_python_index": list(EVENT_TIMING_EARLY_RANGE),
        "event_timing_late_range_python_index": list(EVENT_TIMING_LATE_RANGE),
        "inter_event_short_gap": list(INTER_EVENT_SHORT_GAP),
        "inter_event_long_gap": list(INTER_EVENT_LONG_GAP),
        "strand_seed_offsets": dict(STRAND_SEED_OFFSETS),
    }


@dataclass
class SupplementaryDgChecksRunner:
    """
    Runner for event-based and small-perturbation supplementary data-generating checks.
    """

    config: SimulationConfig

    def __post_init__(self) -> None:
        # Need at least 8 states for subset-based event encoding.
        if self.config.n_states < 8:
            raise ValueError(
                "Supplementary data-generating checks require >= 8 states (subset encoding of 3 events). "
                "Please set n_states=8 and provide 8 state_labels."
            )
        self.rng = np.random.default_rng(self.config.random_seed)
        self.seq_gen = SequenceGenerator(
            state_labels=self.config.state_labels,
            total_length=self.config.total_length,
            random_seed=self.config.random_seed,
        )
        self.dist_computer = DistanceComputer(
            state_labels=self.config.state_labels,
            total_length=self.config.total_length,
            default_norm=self.config.distance_norm,
        )

    def run_event_order(
        self,
        n_replications: int | None = None,
        n_sequences_per_group: int | None = None,
    ) -> Dict[str, Dict[str, float]]:
        return self._run_generic(
            generator=self._generate_event_order_groups,
            strand_name="supplementary_event_order",
            n_replications=n_replications,
            n_sequences_per_group=n_sequences_per_group,
        )

    def run_event_timing(
        self,
        n_replications: int | None = None,
        n_sequences_per_group: int | None = None,
        early_range: Tuple[int, int] = EVENT_TIMING_EARLY_RANGE,
        late_range: Tuple[int, int] = EVENT_TIMING_LATE_RANGE,
    ) -> Dict[str, Dict[str, float]]:
        return self._run_generic(
            generator=lambda n: self._generate_event_timing_groups(
                n=n,
                early_range=early_range,
                late_range=late_range,
            ),
            strand_name="supplementary_event_timing",
            n_replications=n_replications,
            n_sequences_per_group=n_sequences_per_group,
        )

    def run_event_inter_duration(
        self,
        n_replications: int | None = None,
        n_sequences_per_group: int | None = None,
        short_gap: Tuple[int, int] = INTER_EVENT_SHORT_GAP,
        long_gap: Tuple[int, int] = INTER_EVENT_LONG_GAP,
    ) -> Dict[str, Dict[str, float]]:
        return self._run_generic(
            generator=lambda n: self._generate_event_inter_duration_groups(
                n=n,
                short_gap=short_gap,
                long_gap=long_gap,
            ),
            strand_name="supplementary_event_inter_duration",
            n_replications=n_replications,
            n_sequences_per_group=n_sequences_per_group,
        )

    def run_small_perturbation_token(
        self,
        n_replications: int | None = None,
        n_sequences_per_group: int | None = None,
    ) -> Dict[str, Dict[str, float]]:
        return self._run_generic(
            generator=self._generate_small_perturbation_token_groups,
            strand_name="supplementary_small_perturbation_token",
            n_replications=n_replications,
            n_sequences_per_group=n_sequences_per_group,
        )

    def run_small_perturbation_boundary(
        self,
        n_replications: int | None = None,
        n_sequences_per_group: int | None = None,
    ) -> Dict[str, Dict[str, float]]:
        return self._run_generic(
            generator=self._generate_small_perturbation_boundary_groups,
            strand_name="supplementary_small_perturbation_boundary",
            n_replications=n_replications,
            n_sequences_per_group=n_sequences_per_group,
        )

    def _run_generic(
        self,
        generator,
        strand_name: str,
        n_replications: int | None,
        n_sequences_per_group: int | None,
    ) -> Dict[str, Dict[str, float]]:
        methods = self.config.get_distance_methods()
        reps = n_replications if n_replications is not None else self.config.n_replications
        n_per_group = (
            n_sequences_per_group
            if n_sequences_per_group is not None
            else self.config.n_sequences_per_group
        )
        offset = STRAND_SEED_OFFSETS.get(strand_name, 0)

        by_method: Dict[str, List[float]] = {m: [] for m in methods}
        for rep in range(reps):
            seed = self.config.random_seed + offset + rep
            self.rng = np.random.default_rng(seed)
            self.seq_gen.rng = np.random.default_rng(seed)

            group1, group2 = generator(n_per_group)
            all_seqs = group1 + group2
            labels = np.array([0] * len(group1) + [1] * len(group2))

            for method in methods:
                if (
                    method.startswith("LCPspell_expcost_")
                    or method.startswith("RLCPspell_expcost_")
                    or method.startswith("OMspell_expcost_")
                    or method.startswith("OMspellRS_expcost_")
                ):
                    base_method = method.split("_expcost_")[0]
                    expcost = float(method.split("_expcost_")[1])
                    dist_matrix = self.dist_computer.compute_distance_matrix(
                        all_seqs,
                        method=base_method,
                        expcost=expcost,
                        om_indel_cost=self.config.om_indel_cost,
                        om_substitution_cost=self.config.om_substitution_cost,
                        strand_name=strand_name,
                        replication_id=rep + 1,
                    )
                else:
                    dist_matrix = self.dist_computer.compute_distance_matrix(
                        all_seqs,
                        method=method,
                        om_indel_cost=self.config.om_indel_cost,
                        om_substitution_cost=self.config.om_substitution_cost,
                        strand_name=strand_name,
                        replication_id=rep + 1,
                    )
                by_method[method].append(compute_pseudo_r2_single(dist_matrix, labels))

        rep_dicts = [{m: by_method[m][rep_i] for m in methods} for rep_i in range(reps)]
        return aggregate_replications(rep_dicts)

    def _generate_event_order_groups(self, n: int) -> Tuple[List[SpellSeq], List[SpellSeq]]:
        g1, g2 = [], []
        for _ in range(n):
            g1.append(self._events_to_spell_seq(self._sample_events_order_constraint(order="e1_lt_e2")))
            g2.append(self._events_to_spell_seq(self._sample_events_order_constraint(order="e2_lt_e1")))
        return g1, g2

    def _generate_event_timing_groups(
        self,
        n: int,
        early_range: Tuple[int, int],
        late_range: Tuple[int, int],
    ) -> Tuple[List[SpellSeq], List[SpellSeq]]:
        g1, g2 = [], []
        for _ in range(n):
            g1.append(self._events_to_spell_seq(self._sample_events_with_e1_window(early_range)))
            g2.append(self._events_to_spell_seq(self._sample_events_with_e1_window(late_range)))
        return g1, g2

    def _generate_event_inter_duration_groups(
        self,
        n: int,
        short_gap: Tuple[int, int],
        long_gap: Tuple[int, int],
    ) -> Tuple[List[SpellSeq], List[SpellSeq]]:
        g1, g2 = [], []
        for _ in range(n):
            short_events, long_events = self._sample_paired_events_with_gaps(
                short_gap, long_gap
            )
            g1.append(self._events_to_spell_seq(short_events))
            g2.append(self._events_to_spell_seq(long_events))
        return g1, g2

    def _generate_small_perturbation_token_groups(self, n: int) -> Tuple[List[SpellSeq], List[SpellSeq]]:
        base = self.seq_gen.generate_group(n_sequences=n)
        perturbed = [self._perturb_one_token(seq) for seq in base]
        return base, perturbed

    def _generate_small_perturbation_boundary_groups(self, n: int) -> Tuple[List[SpellSeq], List[SpellSeq]]:
        base: List[SpellSeq] = []
        perturbed: List[SpellSeq] = []
        max_attempts = max(1000, 50 * n)
        attempts = 0
        while len(base) < n and attempts < max_attempts:
            attempts += 1
            candidate = self.seq_gen.generate_group(n_sequences=1)[0]
            try:
                perturbed_seq = self._perturb_one_boundary(candidate)
            except ValueError:
                continue
            base.append(candidate)
            perturbed.append(perturbed_seq)
        if len(base) < n:
            raise RuntimeError(
                f"Could generate only {len(base)}/{n} feasible "
                f"boundary perturbations after {attempts} attempts."
            )
        return base, perturbed

    def _sample_events_order_constraint(self, order: str) -> EventTuple:
        t_min, t_max = 2, self.config.total_length - 3
        if t_max - t_min + 1 < 3:
            raise ValueError("total_length too small for event-based constraints.")
        max_attempts = 2000
        for _ in range(max_attempts):
            times = np.sort(self.rng.choice(np.arange(t_min, t_max + 1), size=3, replace=False))
            perm = self.rng.permutation(3)
            event_times = [None, None, None]
            for idx_event, idx_time in enumerate(perm):
                event_times[idx_event] = int(times[idx_time])
            e1, e2, e3 = event_times
            if order == "e1_lt_e2" and e1 < e2:
                return (e1, e2, e3)
            if order == "e2_lt_e1" and e2 < e1:
                return (e1, e2, e3)
        raise RuntimeError(
            f"Could not sample event order constraint {order!r} after {max_attempts} attempts."
        )

    def _sample_events_with_e1_window(self, e1_window: Tuple[int, int]) -> EventTuple:
        lo, hi = e1_window
        T = self.config.total_length
        for _ in range(200):
            e1 = int(self.rng.integers(lo, hi + 1))
            available = [t for t in range(1, T - 1) if t != e1]
            if len(available) < 2:
                continue
            e2, e3 = self.rng.choice(available, size=2, replace=False)
            return (e1, int(e2), int(e3))
        raise RuntimeError("Failed to sample event-timing tuple.")

    def _sample_events_with_gap(self, gap_window: Tuple[int, int]) -> EventTuple:
        short_events, _ = self._sample_paired_events_with_gaps(
            gap_window, gap_window
        )
        return short_events

    def _sample_paired_events_with_gaps(
        self,
        short_gap_window: Tuple[int, int],
        long_gap_window: Tuple[int, int],
    ) -> Tuple[EventTuple, EventTuple]:
        """
        Draw a common e1 support, then apply short versus long e2-e1 gaps on
        the same onset. Event e3 remains independently sampled background
        variation within each group.
        """
        gap_lo_s, gap_hi_s = short_gap_window
        gap_lo_l, gap_hi_l = long_gap_window
        max_gap = max(gap_hi_s, gap_hi_l)
        T = self.config.total_length
        for _ in range(500):
            gap_short = int(self.rng.integers(gap_lo_s, gap_hi_s + 1))
            gap_long = int(self.rng.integers(gap_lo_l, gap_hi_l + 1))
            if gap_short == gap_long:
                continue
            e1 = int(self.rng.integers(2, T - max_gap - 2))
            e2_short = e1 + gap_short
            e2_long = e1 + gap_long
            if e2_long >= T - 2:
                continue
            available_short = [t for t in range(1, T - 1) if t not in (e1, e2_short)]
            available_long = [t for t in range(1, T - 1) if t not in (e1, e2_long)]
            if not available_short or not available_long:
                continue
            e3_short = int(self.rng.choice(available_short))
            e3_long = int(self.rng.choice(available_long))
            return (e1, e2_short, e3_short), (e1, e2_long, e3_long)
        raise RuntimeError("Failed to sample paired inter-event duration tuples.")

    def _events_to_spell_seq(self, events: EventTuple) -> SpellSeq:
        e1, e2, e3 = events
        positions = np.zeros(self.config.total_length, dtype=int)
        for t in range(self.config.total_length):
            code = 0
            if t >= e1:
                code += 1
            if t >= e2:
                code += 2
            if t >= e3:
                code += 4
            positions[t] = code
        labels = [self.config.state_labels[c] for c in positions]
        return self._position_to_spell(labels)

    def _perturb_one_token(self, seq: SpellSeq) -> SpellSeq:
        pos = self.seq_gen.expand_to_position_sequence(*seq)
        idx = int(self.rng.integers(0, self.config.total_length))
        old = pos[idx]
        candidates = [s for s in self.config.state_labels if s != old]
        pos[idx] = self.rng.choice(candidates)
        return self._position_to_spell(pos)

    def _perturb_one_boundary(self, seq: SpellSeq) -> SpellSeq:
        dss, durations = seq
        dss_new = list(dss)
        dur = durations.astype(int).copy()
        if len(dss_new) < 2:
            raise ValueError("Sequence too short for boundary perturbation.")

        moves: List[Tuple[int, int]] = []
        for i in range(len(dur) - 1):
            if dur[i] > 1:
                moves.append((i, i + 1))
            if dur[i + 1] > 1:
                moves.append((i + 1, i))

        if not moves:
            raise ValueError("No feasible boundary perturbation.")

        source, target = moves[int(self.rng.integers(len(moves)))]
        dur[source] -= 1
        dur[target] += 1
        if int(dur.sum()) != self.config.total_length:
            raise RuntimeError(
                f"Boundary perturbation changed total length: sum={int(dur.sum())}."
            )
        return (dss_new, dur)

    def _position_to_spell(self, pos_seq: List[str]) -> SpellSeq:
        if len(pos_seq) != self.config.total_length:
            raise ValueError("Position sequence length mismatch.")
        dss = [pos_seq[0]]
        durations = [1]
        for s in pos_seq[1:]:
            if s == dss[-1]:
                durations[-1] += 1
            else:
                dss.append(s)
                durations.append(1)
        dur = np.array(durations, dtype=int)
        if int(dur.sum()) != self.config.total_length:
            raise RuntimeError(
                f"Position-to-spell conversion: durations sum to {int(dur.sum())}, "
                f"expected {self.config.total_length}."
            )
        return (dss, dur)
