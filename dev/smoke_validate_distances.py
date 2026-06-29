#!/usr/bin/env python3
"""Quick checks before production runs: Elzinga reference + distance-matrix sanity."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_BUNDLE = Path(__file__).resolve().parents[1]
if str(_BUNDLE) not in sys.path:
    sys.path.insert(0, str(_BUNDLE))

from simulation.config import SimulationConfig
from simulation.distance_computer import DistanceComputer
from simulation.elzinga_reference import build_fixed_elzinga_reference

LABELS = ["a", "b", "c", "d", "e"]
T = 20
SEQUENCING_TEMPLATES = [
    ["a", "b", "c", "d", "e"],
    ["e", "d", "c", "b", "a"],
    ["a", "c", "b", "a"],
    ["b", "a", "c", "d", "e"],
    ["a", "b", "c", "e", "d"],
]


def _assert_distance_matrix(m: np.ndarray, n_eval: int, label: str) -> None:
    assert m.shape == (n_eval, n_eval), f"{label}: shape {m.shape} != ({n_eval}, {n_eval})"
    assert np.allclose(m, m.T, atol=1e-12), f"{label}: not symmetric"
    assert np.allclose(np.diag(m), 0.0, atol=1e-12), f"{label}: nonzero diagonal"
    assert np.all(np.isfinite(m)), f"{label}: non-finite values"


def main() -> None:
    ref = build_fixed_elzinga_reference(LABELS, T)
    print("Elzinga reference:", ref)
    assert list(ref[0]) == ["e", "a", "c", "b", "d"]
    assert list(ref[1]) == [4, 4, 4, 4, 4]
    assert list(ref[0]) not in SEQUENCING_TEMPLATES

    cfg = SimulationConfig(total_length=T, n_states=5, n_sequences_per_group=4, n_replications=1)
    seqs = [
        (["a", "b", "c"], np.array([7, 7, 6])),
        (["a", "c", "b"], np.array([6, 8, 6])),
        (["e", "a", "c"], np.array([5, 8, 7])),
    ]
    n_eval = len(seqs)

    for norm in ("none", "auto", "ElzingaStuder"):
        dc = DistanceComputer(cfg.state_labels, cfg.total_length, default_norm=norm)
        for method in ("LCP", "LCPspell"):
            m = dc.compute_distance_matrix(seqs, method, norm=norm, expcost=0.5)
            _assert_distance_matrix(m, n_eval, f"{norm}/{method}")
            print(f"OK {norm}/{method} shape={m.shape}")

    print("All distance-matrix sanity checks passed.")


if __name__ == "__main__":
    main()
