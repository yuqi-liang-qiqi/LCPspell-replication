#!/usr/bin/env python3
"""Invariant tests for timing, directional generators, and evaluation."""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

_BUNDLE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BUNDLE))

from simulation.supplementary_dg_checks.extension_experiments import SupplementaryDgChecksRunner
from simulation.config import SimulationConfig
from simulation.distance_computer import DistanceComputer, validate_spell_sequence
from simulation.evaluation import (
    aggregate_directional_replications,
    aggregate_replications,
    chance_corrected_from_disc,
    compute_directional_pair_scores,
    compute_early_late_directional_scores,
    compute_normalized_paired_contrast,
    compute_paired_win_rate,
    compute_pseudo_r2,
    compute_disc_pseudo_r2,
)
from simulation.framework import SimulationFramework
from simulation.sequence_generator import SequenceGenerator
from simulation.studies.study2_directional_pairs import (
    DirectionalPairStudyMixin,
    EARLY_CONVERGENCE_SUFFIX_RANGE,
    EARLY_DIVERGENCE_PREFIX_RANGE,
)
from simulation.studies.registry import RESULTS_DIRS

PersistentDirectionalMixin = DirectionalPairStudyMixin


def assert_valid_spell_sequence(dss, durations, total_length: int) -> None:
    assert len(dss) == len(durations)
    assert int(durations.sum()) == total_length
    assert np.all(durations >= 1)
    assert all(a != b for a, b in zip(dss, dss[1:]))


def test_timing_preserves_dss_forward_and_reverse():
    gen = SequenceGenerator(["a", "b", "c", "d", "e"], total_length=20, random_seed=0)
    for template in (["a", "b", "c", "d", "e"], ["e", "d", "c", "b", "a"]):
        for t in [6, 8, 12, 14]:
            dss, durations = gen.sample_sequence_with_focal_state_at_time(
                template, "c", t, max_attempts=500
            )
            assert dss == template
            assert durations.sum() == 20
            assert np.all(durations >= 1)
            expanded = gen.expand_to_position_sequence(dss, durations)
            assert expanded[t] == "c"


def test_balanced_chance_corrected_matches_disc_affine_transform():
    dist = np.array(
        [
            [0.0, 1.0, 3.0, 4.0],
            [1.0, 0.0, 2.0, 3.0],
            [3.0, 2.0, 0.0, 1.0],
            [4.0, 3.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    labels = np.array([0, 0, 1, 1])

    r2_cc = compute_pseudo_r2(dist, labels)
    r2_disc = compute_disc_pseudo_r2(dist, labels)
    expected = chance_corrected_from_disc(r2_disc, n=4)
    assert np.isclose(r2_cc, expected)


def test_negative_pseudo_r2_not_clipped():
    dist = np.array(
        [
            [0.0, 100.0, 1.0, 1.0],
            [100.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 100.0],
            [1.0, 1.0, 100.0, 0.0],
        ],
        dtype=float,
    )
    labels = np.array([0, 0, 1, 1])
    r2_cc = compute_pseudo_r2(dist, labels)
    assert r2_cc < -1.0


def test_normalized_paired_contrast_is_scale_invariant():
    assert np.isclose(
        compute_normalized_paired_contrast(20.0, 10.0),
        compute_normalized_paired_contrast(200.0, 100.0),
    )


def test_directional_pair_scores_divergence_and_convergence():
    d_late = np.array([2.0, 2.5, 3.0])
    d_early = np.array([5.0, 6.0, 7.0])
    div_contrast, div_win = compute_directional_pair_scores(
        d_late, d_early, "divergence"
    )
    assert 0.0 < div_contrast < 1.0
    assert div_win > 0

    # Low-level API: convergence expects d_signal < d_control (late < early here).
    conv_contrast, conv_win = compute_directional_pair_scores(
        d_early, d_late, "convergence"
    )
    assert 0.0 < conv_contrast < 1.0
    assert conv_win > 0


def test_early_late_convergence_score_positive_when_early_pair_is_closer():
    d_early = np.array([1.0, 2.0, 3.0])
    d_late = np.array([4.0, 5.0, 6.0])
    score, win = compute_early_late_directional_scores(
        d_early, d_late, "convergence"
    )
    assert score > 0
    assert win > 0


def test_early_late_divergence_score_positive_when_early_pair_is_farther():
    d_early = np.array([5.0, 6.0, 7.0])
    d_late = np.array([2.0, 2.5, 3.0])
    score, win = compute_early_late_directional_scores(
        d_early, d_late, "divergence"
    )
    assert score > 0
    assert win > 0


def _synthetic_early_close_late_far_dist_matrix(
    sequences,
    method,
    strand_name,
    replication_id,
):
    """Early focal pairs distance 1.0; late focal pairs 3.0 (convergence-positive)."""
    n = len(sequences)
    dist = np.full((n, n), 2.0, dtype=float)
    np.fill_diagonal(dist, 0.0)
    for base in range(0, n, 4):
        dist[base, base + 1] = dist[base + 1, base] = 1.0
        dist[base + 2, base + 3] = dist[base + 3, base + 2] = 3.0
    return dist


def test_framework_directional_evaluation_uses_early_late_scores():
    config = SimulationConfig(
        n_sequences_per_group=4,
        n_replications=1,
        verbose=False,
    )
    fw = SimulationFramework(config)
    matched_draws = []
    for _ in range(4):
        early = fw._sample_calendar_convergence_pair(EARLY_CONVERGENCE_SUFFIX_RANGE)
        late = fw._sample_calendar_convergence_pair((3, 5))
        matched_draws.append((early, late))

    with patch.object(
        fw,
        "_compute_distance_matrix_for_method",
        side_effect=_synthetic_early_close_late_far_dist_matrix,
    ):
        results = fw._evaluate_directional_early_late_for_strand(
            matched_draws,
            contrast="convergence",
            strand_name="test_convergence",
            replication_id=0,
        )
    hamming_score = results["HAM"]["contrast"]
    assert hamming_score > 0


def test_paired_win_rate_random_baseline():
    deltas = np.array([1.0, -1.0, 1.0, -1.0])
    assert np.isclose(compute_paired_win_rate(deltas), 0.0)


def test_paired_win_rate_near_zero_positive_is_not_double_counted():
    score = compute_paired_win_rate(np.array([1e-12]))
    assert -1.0 <= score <= 1.0


def test_directional_pair_scores_reject_invalid_inputs():
    with pytest.raises(ValueError, match="non-finite"):
        compute_directional_pair_scores(
            np.array([0.0, np.nan]),
            np.array([1.0, 2.0]),
            "divergence",
        )
    with pytest.raises(ValueError, match="non-negative"):
        compute_directional_pair_scores(
            np.array([-0.1, 1.0]),
            np.array([1.0, 2.0]),
            "divergence",
        )


def test_aggregate_directional_replications():
    reps = [
        {"LCP": {"contrast": 0.2, "win": 0.4}, "RLCP": {"contrast": 0.1, "win": 0.2}},
        {"LCP": {"contrast": 0.4, "win": 0.6}, "RLCP": {"contrast": 0.3, "win": 0.4}},
    ]
    agg = aggregate_directional_replications(reps)
    assert np.isclose(agg["LCP"]["mean"], 0.3)
    assert np.isclose(agg["LCP"]["win_mean"], 0.5)


def _symmetric_dist(n: int, off_diag: float = 1.0) -> np.ndarray:
    dist = np.full((n, n), off_diag, dtype=float)
    np.fill_diagonal(dist, 0.0)
    return dist


def test_third_group_raises():
    dist = _symmetric_dist(6)
    labels = np.array([0, 0, 1, 1, 2, 2])
    with pytest.raises(ValueError, match="exactly two groups"):
        compute_pseudo_r2(dist, labels)


def test_unequal_groups_raises():
    dist = _symmetric_dist(5)
    labels = np.array([0, 0, 0, 1, 1])
    with pytest.raises(ValueError, match="balanced two-group"):
        compute_pseudo_r2(dist, labels)


def test_asymmetric_matrix_raises():
    dist = np.array([[0.0, 1.0], [2.0, 0.0]])
    labels = np.array([0, 1])
    with pytest.raises(ValueError, match="symmetric"):
        compute_pseudo_r2(dist, labels)


def test_aggregate_replications_rejects_inconsistent_keys():
    with pytest.raises(RuntimeError, match="inconsistent method keys"):
        aggregate_replications([
            {"LCP": 0.1, "OM": 0.2},
            {"LCP": 0.3},
        ])


def test_supplementary_boundary_generator_runs():
    config = SimulationConfig(n_states=8, n_sequences_per_group=4, n_replications=1)
    config.state_labels = [f"s{i}" for i in range(8)]
    runner = SupplementaryDgChecksRunner(config)
    base, perturbed = runner._generate_small_perturbation_boundary_groups(3)
    assert len(base) == 3
    assert len(perturbed) == 3
    for seq in base + perturbed:
        assert_valid_spell_sequence(*seq, config.total_length)


def test_distance_computer_rejects_invalid_spell_sequences():
    dc = DistanceComputer(["a", "b", "c"], total_length=20)
    with pytest.raises(ValueError, match="adjacent duplicate"):
        validate_spell_sequence(
            ["a", "a", "b"], np.array([5, 5, 10]), total_length=20, state_labels=["a", "b", "c"]
        )
    with pytest.raises(ValueError, match="positive"):
        validate_spell_sequence(
            ["a", "b"], np.array([0, 20]), total_length=20, state_labels=["a", "b", "c"]
        )
    with pytest.raises(ValueError, match="DSS length"):
        dc.sequences_to_seqdata([(["a", "b"], np.array([10]))])


def test_expcost_grid_rejects_rounding_collision():
    with pytest.raises(ValueError, match="two decimal places"):
        SimulationConfig(expcost_values=[0.101, 0.104])

    with pytest.raises(ValueError, match="duplicate"):
        SimulationConfig(expcost_values=[0.10, 0.1])


def test_distance_norm_validation():
    with pytest.raises(ValueError, match="distance_norm"):
        SimulationConfig(distance_norm="typo")


def test_directional_duration_mode_validation():
    with pytest.raises(ValueError, match="directional_duration_mode"):
        SimulationConfig(directional_duration_mode="invalid")


def test_empty_state_labels_raise():
    with pytest.raises(ValueError, match="at least one state"):
        SimulationConfig(state_labels=[])


def test_fractional_durations_raise():
    with pytest.raises(ValueError, match="integers"):
        validate_spell_sequence(
            ["a", "b"],
            np.array([1.5, 18.5]),
            total_length=20,
            state_labels=["a", "b"],
        )


def test_expand_rejects_mismatched_dss_duration_lengths():
    gen = SequenceGenerator(["a", "b"], total_length=20, random_seed=0)
    with pytest.raises(ValueError, match="DSS length"):
        gen.expand_to_position_sequence(["a", "b", "c"], np.array([10, 10]))


def test_calendar_divergence_pair_structure():
    config = SimulationConfig(verbose=False)
    fw = SimulationFramework(config)
    fw.seq_generator.rng = np.random.default_rng(2)

    seq_a, seq_b = fw._sample_calendar_divergence_pair((5, 5))
    pos_a = fw.seq_generator.expand_to_position_sequence(*seq_a)
    pos_b = fw.seq_generator.expand_to_position_sequence(*seq_b)

    prefix_len = 5
    assert pos_a[:prefix_len] == pos_b[:prefix_len]
    assert all(a != b for a, b in zip(pos_a[prefix_len:], pos_b[prefix_len:]))


def test_calendar_convergence_pair_structure():
    config = SimulationConfig(verbose=False)
    fw = SimulationFramework(config)
    fw.seq_generator.rng = np.random.default_rng(3)

    seq_a, seq_b = fw._sample_calendar_convergence_pair((15, 15))
    pos_a = fw.seq_generator.expand_to_position_sequence(*seq_a)
    pos_b = fw.seq_generator.expand_to_position_sequence(*seq_b)

    suffix_len = 15
    assert pos_a[:-suffix_len] != pos_b[:-suffix_len]
    assert pos_a[-suffix_len:] == pos_b[-suffix_len:]


def test_spell_order_divergence_pair_structure():
    config = SimulationConfig(verbose=False)
    fw = SimulationFramework(config)
    fw.seq_generator.rng = np.random.default_rng(4)

    seq_a, seq_b = fw._sample_spell_order_divergence_pair(1)
    prefix_len = 1
    assert seq_a[0][:prefix_len] == seq_b[0][:prefix_len]
    assert all(
        a != b for a, b in zip(seq_a[0][prefix_len:], seq_b[0][prefix_len:])
    )
    assert_valid_spell_sequence(*seq_a, config.total_length)
    assert_valid_spell_sequence(*seq_b, config.total_length)


def test_spell_order_convergence_pair_structure():
    config = SimulationConfig(verbose=False)
    fw = SimulationFramework(config)
    fw.seq_generator.rng = np.random.default_rng(5)

    seq_a, seq_b = fw._sample_spell_order_convergence_pair(4)
    suffix_len = 4
    assert seq_a[0][:-suffix_len] != seq_b[0][:-suffix_len]
    assert seq_a[0][-suffix_len:] == seq_b[0][-suffix_len:]
    assert_valid_spell_sequence(*seq_a, config.total_length)
    assert_valid_spell_sequence(*seq_b, config.total_length)


def test_directional_generators_produce_valid_pairs():
    config = SimulationConfig(n_sequences_per_group=6, n_replications=1, verbose=False)
    fw = SimulationFramework(config)
    fw.seq_generator.rng = np.random.default_rng(0)

    for sample_fn in (
        lambda: fw._sample_calendar_divergence_pair(EARLY_DIVERGENCE_PREFIX_RANGE),
        lambda: fw._sample_calendar_convergence_pair(EARLY_CONVERGENCE_SUFFIX_RANGE),
        lambda: fw._sample_spell_order_divergence_pair(1),
        lambda: fw._sample_spell_order_convergence_pair(4),
    ):
        for _ in range(6):
            pair = sample_fn()
            for seq in pair:
                fw._assert_total_duration(seq[0], seq[1], context="pair")


def _position_hamming(fw: SimulationFramework, seq_a, seq_b) -> int:
    pos_a = fw.seq_generator.expand_to_position_sequence(*seq_a)
    pos_b = fw.seq_generator.expand_to_position_sequence(*seq_b)
    return int(sum(a != b for a, b in zip(pos_a, pos_b)))


def test_nested_calendar_divergence_shares_late_tail():
    config = SimulationConfig(verbose=False)
    fw = SimulationFramework(config)
    fw.seq_generator.rng = np.random.default_rng(21)

    (ea, eb), (la, lb) = fw._sample_nested_calendar_divergence_draw(
        EARLY_DIVERGENCE_PREFIX_RANGE, (14, 17)
    )
    pos_ea = fw.seq_generator.expand_to_position_sequence(*ea)
    pos_la = fw.seq_generator.expand_to_position_sequence(*la)
    pos_eb = fw.seq_generator.expand_to_position_sequence(*eb)
    pos_lb = fw.seq_generator.expand_to_position_sequence(*lb)

    L_late = next(
        i
        for i in range(config.total_length)
        if pos_la[i] != pos_lb[i]
    )
    assert pos_ea[L_late:] == pos_la[L_late:]
    assert pos_eb[L_late:] == pos_lb[L_late:]


def test_nested_spell_order_matched_durations_shared():
    config = SimulationConfig(verbose=False, directional_duration_mode="matched")
    fw = SimulationFramework(config)
    fw.seq_generator.rng = np.random.default_rng(22)

    (ea, eb), (la, lb) = fw._sample_nested_spell_order_divergence_draw(1, 3)
    assert np.array_equal(ea[1], eb[1])
    assert np.array_equal(ea[1], la[1])
    assert np.array_equal(ea[1], lb[1])


def test_early_vs_late_calendar_divergence_hamming():
    """Early divergence pairs should be farther apart than late divergence pairs."""
    config = SimulationConfig(verbose=False)
    fw = SimulationFramework(config)
    fw.seq_generator.rng = np.random.default_rng(99)

    early_dists, late_dists = [], []
    for _ in range(200):
        (ea, eb), (la, lb) = fw._sample_nested_calendar_divergence_draw(
            EARLY_DIVERGENCE_PREFIX_RANGE, (14, 17)
        )
        early_dists.append(_position_hamming(fw, ea, eb))
        late_dists.append(_position_hamming(fw, la, lb))

    assert np.mean(early_dists) > np.mean(late_dists) + 1.0


def test_early_vs_late_calendar_convergence_hamming():
    """Early convergence pairs should be more similar than late convergence pairs."""
    config = SimulationConfig(verbose=False)
    fw = SimulationFramework(config)
    fw.seq_generator.rng = np.random.default_rng(100)

    early_dists, late_dists = [], []
    for _ in range(200):
        (ea, eb), (la, lb) = fw._sample_nested_calendar_convergence_draw(
            EARLY_CONVERGENCE_SUFFIX_RANGE, (3, 5)
        )
        early_dists.append(_position_hamming(fw, ea, eb))
        late_dists.append(_position_hamming(fw, la, lb))

    assert np.mean(early_dists) < np.mean(late_dists) - 1.0


def test_early_vs_late_divergence_contrast_not_saturated():
    """Normalized contrast should discriminate early vs late, not saturate at 1."""
    config = SimulationConfig(verbose=False)
    fw = SimulationFramework(config)
    fw.seq_generator.rng = np.random.default_rng(101)

    d_early, d_late = [], []
    for _ in range(100):
        (ea, eb), (la, lb) = fw._sample_nested_calendar_divergence_draw(
            EARLY_DIVERGENCE_PREFIX_RANGE, (14, 17)
        )
        d_early.append(float(_position_hamming(fw, ea, eb)))
        d_late.append(float(_position_hamming(fw, la, lb)))

    score, _ = compute_early_late_directional_scores(
        np.asarray(d_early), np.asarray(d_late), "divergence"
    )
    assert 0.0 < score < 1.0


def test_build_calendar_template_helpers():
    pos1, pos2 = PersistentDirectionalMixin.build_calendar_divergence_templates(
        ["a", "a"], ["b", "b"], ["c", "c"]
    )
    assert pos1 == ["a", "a", "b", "b"]
    assert pos2 == ["a", "a", "c", "c"]

    pos3, pos4 = PersistentDirectionalMixin.build_calendar_convergence_templates(
        ["a", "b"], ["c", "d"], ["x", "y"]
    )
    assert pos3 == ["a", "b", "x", "y"]
    assert pos4 == ["c", "d", "x", "y"]

    with pytest.raises(ValueError, match="differ at every position"):
        PersistentDirectionalMixin.build_calendar_divergence_templates(
            ["a"], ["b"], ["b"]
        )


def test_inter_event_duration_paired_sampling_aligns_e1_marginals():
    config = SimulationConfig(
        total_length=20,
        n_states=8,
        state_labels=list("abcdefgh"),
        verbose=False,
    )
    runner = SupplementaryDgChecksRunner(config=config)
    runner.rng = np.random.default_rng(7)

    e1_short, e1_long = [], []
    for _ in range(500):
        short_events, long_events = runner._sample_paired_events_with_gaps(
            (1, 3), (6, 8)
        )
        e1_short.append(short_events[0])
        e1_long.append(long_events[0])

    assert abs(np.mean(e1_short) - np.mean(e1_long)) < 0.75


@pytest.mark.parametrize(
    "duration_mode",
    ["matched", "shared_spell_mismatch_compensated", "background_noise"],
)
def test_spell_order_generators_all_duration_modes(duration_mode):
    config = SimulationConfig(
        verbose=False,
        directional_duration_mode=duration_mode,
    )
    fw = SimulationFramework(config)
    fw.seq_generator.rng = np.random.default_rng(7)

    for sample_fn in (
        lambda: fw._sample_spell_order_divergence_pair(1),
        lambda: fw._sample_spell_order_convergence_pair(4),
    ):
        pair = sample_fn()
        for seq in pair:
            fw._assert_total_duration(seq[0], seq[1], context="pair")


if __name__ == "__main__":
    test_timing_preserves_dss_forward_and_reverse()
    test_calendar_divergence_pair_structure()
    test_calendar_convergence_pair_structure()
    test_directional_generators_produce_valid_pairs()
    print("All dev tests passed.")
