"""
Fixed synthetic reference trajectory for Elzinga--Studer normalization (appendix).

The reference is defined independently of the two simulated groups and held
constant across replications. State order is a fixed permutation (not the
alphabet cycle used in sequencing group templates). It is appended only during
distance computation and excluded from pseudo-R² evaluation.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

SpellSeq = Tuple[List[str], np.ndarray]

# Fixed seed for the pre-specified state-order permutation (reproducible across runs).
ELZINGA_REFERENCE_PERMUTATION_SEED = 271828


def _fixed_state_order_indices(n_states: int) -> np.ndarray:
    rng = np.random.default_rng(ELZINGA_REFERENCE_PERMUTATION_SEED)
    return rng.permutation(n_states)


def build_fixed_elzinga_reference(
    state_labels: List[str],
    total_length: int,
) -> SpellSeq:
    """
    Build a deterministic synthetic reference spell sequence.

    The state order is a fixed permutation independent of simulation group
    templates. Spell lengths are distributed as evenly as possible across spells.

    For T=20 and state_labels ``['a','b','c','d','e']`` this yields
    ``(['e','a','c','b','d'], [4,4,4,4,4])`` (order from seed 271828).
    """
    n = len(state_labels)
    if n < 1:
        raise ValueError("state_labels must be non-empty")
    if total_length < n:
        raise ValueError(
            f"total_length ({total_length}) must be at least n_states ({n}) "
            "for the fixed Elzinga reference."
        )

    order = _fixed_state_order_indices(n)
    dss = [state_labels[i] for i in order]

    base = total_length // n
    rem = total_length % n
    durations = [base + (1 if i < rem else 0) for i in range(n)]
    dur = np.array(durations, dtype=int)

    if int(dur.sum()) != total_length:
        raise RuntimeError("Elzinga reference durations do not sum to total_length.")

    return (dss, dur)
