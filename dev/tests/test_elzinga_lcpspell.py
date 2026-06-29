"""Regression: ElzingaStuder must work with LCPspell (appendix robustness run)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_BUNDLE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BUNDLE))

from simulation.config import SimulationConfig
from simulation.distance_computer import DistanceComputer


@pytest.fixture
def spell_sequences():
    return [
        (["a", "b", "c"], np.array([7, 7, 6])),
        (["a", "c", "b"], np.array([6, 8, 6])),
        (["e", "a", "c"], np.array([5, 8, 7])),
    ]


def test_elzinga_lcpspell_distance_matrix(spell_sequences):
    cfg = SimulationConfig(total_length=20, n_states=5, n_sequences_per_group=4, n_replications=1)
    dc = DistanceComputer(cfg.state_labels, cfg.total_length, default_norm="ElzingaStuder")
    n_eval = len(spell_sequences)

    dist = dc.compute_distance_matrix(
        spell_sequences,
        "LCPspell",
        norm="ElzingaStuder",
        expcost=0.5,
    )

    assert dist.shape == (n_eval, n_eval)
    assert np.allclose(dist, dist.T, atol=1e-12)
    assert np.allclose(np.diag(dist), 0.0, atol=1e-12)
    assert np.all(np.isfinite(dist))
